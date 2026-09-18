"""Credit to https://github.com/albanie/slurm_gpustat with minor modifications for web app.

A simple tool for summarising GPU statistics on a slurm cluster.

The tool can be used in two ways:
1. To simply query the current usage of GPUs on the cluster.
2. To launch a daemon which will log usage over time.  This can then later be queried
   to provide simple usage statistics.
"""

import os
import re
import ast
import sys
import time
import threading
import atexit
import signal
import argparse
import functools
import subprocess
from typing import Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Same reason as in app.py: the login node is close to its strict-overcommit
# limit, and a 40-thread OpenBLAS pool is both useless here and large enough to
# push allocations over it.  Must run before numpy is imported.
for _blas_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_blas_var, "1")

import numpy as np
import colored
import humanize
import humanfriendly as hf
from beartype import beartype
from django.utils.functional import lazy


# SLURM states which indicate that the node is not available for submitting jobs
INACCESSIBLE = {"drain*", "down*", "drng", "drain", "down"}
INACCESSIBLE_KEYWORDS = ("down", "drain", "drng", "resv", "inval", "plnd", "pow_up", "reboot")
INTERACTIVE_CMDS = {"bash", "zsh", "sh"}
# How hard parse_cmd tries before letting a SLURM query fail, see its docstring.
SLURM_RETRIES = 3
SLURM_RETRY_DELAY = 1.0


def is_inaccessible(state: str) -> bool:
    state_lower = state.lower().rstrip("*~$#%")
    if state_lower in INACCESSIBLE:
        return True
    return any(kw in state_lower for kw in INACCESSIBLE_KEYWORDS)


class Daemon:
    """A Generic linux daemon base class for python 3.x.

    This code is a Python3 port of Sander Marechal's Daemon module:
    http://web.archive.org/web/20131017130434/http://www.jejik.com/articles/
    2007/02/a_simple_unix_linux_daemon_in_python/

    It's a little difficult to credit the author of the Python3 port, since the code was
    published anonymously. The original can be found here:
    http://web.archive.org/web/20131101191715/http://www.jejik.com/files/examples/
    daemon3x.py
    """

    def __init__(self, pidfile):
        self.pidfile = pidfile

    def daemonize(self):
        """Deamonize class. UNIX double fork mechanism."""
        try:
            pid = os.fork()
            if pid > 0:
                # exit first parent
                sys.exit(0)
        except OSError as err:
            sys.stderr.write('fork #1 failed: {0}\n'.format(err))
            sys.exit(1)

        # decouple from parent environment
        os.chdir('/')
        os.setsid()
        os.umask(0)

        # do second fork
        try:
            pid = os.fork()
            if pid > 0:
                # exit from second parent
                sys.exit(0)
        except OSError as err:
            sys.stderr.write('fork #2 failed: {0}\n'.format(err))
            sys.exit(1)
        
        # redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        si = open(os.devnull, 'r')
        so = open(os.devnull, 'a+')
        se = open(os.devnull, 'a+')

        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())

        # write pidfile
        atexit.register(self.delpid)
        signal.signal(signal.SIGTERM, lambda signum, stack_frame: exit())

        pid = str(os.getpid())
        with open(self.pidfile, 'w+') as f:
            f.write(pid + '\n')

    def delpid(self):
        os.remove(self.pidfile)

    def start(self):
        """Start the daemon."""

        # Check for a pidfile to see if the daemon already runs
        try:
            with open(self.pidfile, 'r') as pf:
                pid = int(pf.read().strip())
        except IOError:
            pid = None

        if pid:
            message = "pidfile {0} already exists. Is the daemon already running?\n"
            sys.stderr.write(message.format(self.pidfile))
            sys.exit(1)

        # Start the daemon
        self.daemonize()
        self.run()

    def stop(self):
        """Stop the daemon."""

        # Get the pid from the pidfile
        try:
            with open(self.pidfile, 'r') as pf:
                pid = int(pf.read().strip())
        except IOError:
            pid = None

        if not pid:
            message = "pidfile {0} does not exist. Daemon not running?\n"
            sys.stderr.write(message.format(self.pidfile))
            return  # not an error in a restart

        # Try killing the daemon process
        try:
            while 1:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.1)
        except OSError as err:
            e = str(err.args)
            if e.find("No such process") > 0:
                if os.path.exists(self.pidfile):
                    os.remove(self.pidfile)
            else:
                print(str(err.args))
                sys.exit(1)

    def restart(self):
        """Restart the daemon."""
        self.stop()
        self.start()

    def run(self):
        """You should override this method when you subclass Daemon.
        It will be called after the process has been daemonized by
        start() or restart()."""
        raise NotImplementedError("Must override this class")


class GPUStatDaemon(Daemon):
    """A lightweight daemon which intermittently logs gpu usage to a text file.
    """
    timestamp_format = "%Y-%m-%d_%H:%M:%S"

    def __init__(self, pidfile, log_path, log_interval):
        """Create the daemon.

        Args:
            pidfile (str): the location of the daemon pid file.
            log_path (str): the location where the historical log will be stored.
            log_interval (int): the time interval (in seconds) at which gpu usage will
                be stored to the log.
        """
        Path(pidfile).parent.mkdir(exist_ok=True, parents=True)
        super().__init__(pidfile=pidfile)
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)
        self.log_interval = log_interval
        self.log_path = log_path

    def serialize_usage(self, usage):
        """Convert data structure into an appropriate string for serialization.

        Args:
            usage (a dict-like structure): a data-structure which has the form of a
                dictionary, but may contain variants with length string representations
                (e.g. defaultdict, OrderedDict etc.)

        Returns:
            (str): a string representation of the usage data strcture.
        """
        for user, gpu_dict in usage.items():
            for key, subdict in gpu_dict.items():
                usage[user][key] = dict(subdict)
        usage = dict(usage)
        return usage.__repr__()

    @staticmethod
    def deserialize_usage(log_path):
        """Parse the `usage` data structure by reading in the contents of the text-based
        log file and deserializing.

        Args:
            log_path (str): the location of the log file.

        Returns:
            (list[dict]): a list of dicts, where each dict contains the time stamp
                associated with a set of usage statistics, together with the statistics
                themselves.
        """
        if not Path(log_path).exists():
            raise ValueError("No historical log found.  Did you start the daemon?")
        with open(log_path, "r") as f:
            rows = f.read().splitlines()
        data = []
        for row in rows:
            ts, usage = row.split(maxsplit=1)
            dt = datetime.strptime(ts, GPUStatDaemon.timestamp_format)
            usage = ast.literal_eval(usage)
            data.append({"timestamp": dt, "usage": usage})
        return data

    def run(self):
        """Run the daemon - will intermittently log gpu usage to disk.
        """
        while True:
            resources = parse_all_gpus()
            usage = gpu_usage(resources)
            log_row = self.serialize_usage(usage)
            timestamp = datetime.now().strftime(GPUStatDaemon.timestamp_format)
            with open(self.log_path, "a") as f:
                f.write(f"{timestamp} {log_row}\n")
            time.sleep(self.log_interval)


def historical_summary(data):
    """Print a short summary of the historical gpu usage logged by the daemon.

    Args:
        data (list): the data structure deserialized from the daemon log file (this is
            the output of the GPUStatDaemon.deserialize_usage() function.)
    """
    first_ts, last_ts = data[0]["timestamp"], data[-1]["timestamp"]
    print(f"Historical data contains {len(data)} samples ({first_ts} to {last_ts})")
    latest_usage = data[-1]["usage"]
    users, gpu_types = set(), set()
    for user, resources in latest_usage.items():
        users.add(user)
        gpu_types.update(set(resources.keys()))
    history = {}
    for row in data:
        for user, subdict in row["usage"].items():
            if user not in history:
                history[user] = {gpu_type: [] for gpu_type in gpu_types}
            type_counts = {key: sum(val.values()) for key, val in subdict.items()}
            for gpu_type in gpu_types:
                history[user][gpu_type].append(type_counts.get(gpu_type, 0))

    for user, subdict in history.items():
        print(f"GPU usage for {user}:")
        total = 0
        for gpu_type, counts in subdict.items():
            counts = np.array(counts)
            if counts.sum() == 0:
                continue
            print(f"{gpu_type:5s} > avg: {int(counts.mean())}, max: {np.max(counts)}")
            total += counts.mean()
        print(f"total > avg: {int(total)}\n")


def split_node_str(node_str):
    """Split SLURM node specifications into node_specs. Here a node_spec defines a range
    of nodes that share the same naming scheme (and are grouped together using square
    brackets).   E.g. 'node[1-3,4,6-9]' represents a single node_spec.

    Examples:
       A `node_str` of the form 'node[001-003]' will be returned as a single element
           list: ['node[001-003]']
       A `node_str` of the form 'node[001-002],node004' will be split into
           ['node[001-002]', 'node004']

    Args:
        node_str (str): a SLURM-formatted list of nodes

    Returns:
        (list[str]): SLURM node specs.
    """
    node_str = node_str.strip()
    breakpoints, stack = [0], []
    for ii, char in enumerate(node_str):
        if char == "[":
            stack.append(char)
        elif char == "]":
            stack.pop()
        elif not stack and char == ",":
            breakpoints.append(ii + 1)
    end = len(node_str) + 1
    return [node_str[i: j - 1] for i, j in zip(breakpoints, breakpoints[1:] + [end])]


def parse_node_names(node_str):
    """Parse the node list produced by the SLURM tools into separate node names.

    Examples:
       A slurm `node_str` of the form 'node[001-003]' will be split into a list of the
           form ['node001', 'node002', 'node003'].
       A `node_str` of the form 'node[001-002],node004' will be split into
           ['node001', 'node002', 'node004']

    Args:
        node_str (str): a SLURM-formatted list of nodes

    Returns:
        (list[str]): a list of separate node names.
    """
    names = []
    node_specs = split_node_str(node_str)
    for node_spec in node_specs:
        if "[" not in node_spec:
            names.append(node_spec)
        else:
            head, tail = node_spec.index("["), node_spec.index("]")
            prefix = node_spec[:head]
            subspecs = node_spec[head + 1:tail].split(",")
            for subspec in subspecs:
                if "-" not in subspec:
                    subnames = [f"{prefix}{subspec}"]
                else:
                    start, end = subspec.split("-")
                    num_digits = len(start)
                    subnames = [f"{prefix}{str(x).zfill(num_digits)}"
                                for x in range(int(start), int(end) + 1)]
                names.extend(subnames)
    return names


# A full page refresh fetches every panel at once and several of them ask SLURM
# the same question: the same squeue, sinfo and scontrol lines were being run
# three and four times over, about 1.3s of the roughly 7s a refresh spends in
# subprocesses.  Hold each answer briefly, and make callers that ask while one is
# still in flight wait for it rather than start a second copy, which is the case
# a plain expiry would miss when eight panels start together.
CMD_CACHE_SECONDS = int(os.environ.get("RIVANNA_CMD_CACHE_SECONDS", "10"))
# sstat readings are subtracted from one another to turn counters into a rate, so
# two calls a few seconds apart have to be two different readings.
UNCACHEABLE_COMMANDS = ("sstat",)
_cmd_cache = {}
_cmd_cache_guard = threading.Lock()
_cmd_locks = {}


def _cmd_is_cacheable(cmd):
    if CMD_CACHE_SECONDS <= 0:
        return False
    head = cmd.split()[0] if cmd.split() else ""
    return head not in UNCACHEABLE_COMMANDS


def _run_slurm_cmd(cmd, split, retries, delay):
    """Parse the output of a shell command...
     and if split set to true: split into a list of strings, one per line of output.

    Two transient failures are worth riding out rather than propagating.  A busy
    slurmctld sometimes answers a perfectly valid query with a truncated RPC
    ("Malformed RPC of type RESPONSE_JOB_INFO", "Header lengths are longer than
    data received"), which makes the client exit non-zero for a second or two.
    And the login node runs with strict overcommit while sitting near its commit
    limit, so starting the child at all can fail with ENOMEM.  Every command sent
    through here is a read-only query, so retry a few times before giving up
    rather than letting one bad moment take out a page.

    Args:
        cmd (str): the shell command to be executed.
        split (bool): whether to split the output per line
        retries (int): how many times to run the command before giving up.
        delay (float): seconds to wait between attempts.
    Returns:
        (list[str]): the strings from each output line.
    Raises:
        subprocess.CalledProcessError: if the command ran and failed every time.
        OSError: if the command could not be started at all.
    """
    for attempt in range(1, retries + 1):
        try:
            output = subprocess.check_output(
                cmd, shell=True, stderr=subprocess.PIPE
            ).decode("utf-8")
            break
        except (subprocess.CalledProcessError, OSError) as exc:
            if isinstance(exc, subprocess.CalledProcessError):
                reason = " ".join((exc.stderr or b"").decode("utf-8", "replace").split())
                detail = f"exit {exc.returncode}: {reason or 'no stderr'}"
            else:
                detail = f"could not be started: {exc}"
            if attempt == retries:
                if retries > 1:
                    print(f"Warning: `{cmd}` still failing after {retries} attempts ({detail})")
                raise
            print(
                f"Warning: `{cmd}` failed ({detail}); "
                f"retrying in {delay}s ({attempt}/{retries - 1})"
            )
            time.sleep(delay)
    if split:
        output = [x for x in output.split("\n") if x]
    return output


def parse_cmd(cmd, split=True, retries=SLURM_RETRIES, delay=SLURM_RETRY_DELAY):
    """Run a SLURM query, reusing an answer from the last CMD_CACHE_SECONDS.

    See :func:`_run_slurm_cmd` for what actually runs and why it retries.
    """
    if not _cmd_is_cacheable(cmd):
        return _run_slurm_cmd(cmd, split, retries, delay)

    key = (cmd, split)

    def _fresh():
        """The cached answer if it is still young enough, else None."""
        with _cmd_cache_guard:
            entry = _cmd_cache.get(key)
        if entry is None or time.monotonic() - entry[0] >= CMD_CACHE_SECONDS:
            return None
        # Callers get their own list, so one of them sorting it in place cannot
        # reshuffle what the next caller reads out of the cache.
        return list(entry[1]) if isinstance(entry[1], list) else entry[1]

    hit = _fresh()
    if hit is not None:
        return hit

    with _cmd_cache_guard:
        lock = _cmd_locks.setdefault(key, threading.Lock())
    with lock:
        # Whoever held this lock has just filled the cache, so look again before
        # running a second copy of their command.
        hit = _fresh()
        if hit is not None:
            return hit
        output = _run_slurm_cmd(cmd, split, retries, delay)
        with _cmd_cache_guard:
            _cmd_cache[key] = (time.monotonic(), output)
    return list(output) if isinstance(output, list) else output


@beartype
def node_states(partition: Optional[str] = None) -> dict:
    """Query SLURM for the state of each managed node.

    Args:
        partition: the partition/queue (or multiple, comma separated) of interest.
            By default None, which queries all available partitions.

    Returns:
        a mapping between node names and SLURM states.
    """
    cmd = "sinfo --noheader"
    if partition:
        cmd += f" --partition={partition}"
    else:
        cmd += " -a"
    rows = parse_cmd(cmd)
    states = {}
    for row in rows:
        tokens = row.split()
        state, names = tokens[4], tokens[5]
        node_names = parse_node_names(names)
        states.update({name: state for name in node_names})
    return states


@beartype
def get_gpu_partitions(keywords=['gpu', 'ddp']) -> list:
    """Query SLURM for the supported partitions.

    Args:
        keywords: the keywords to indicate gpu partitions (comma separated) of interest.

    Returns:
        a list of requested SLURM partitions.
    """
    cmd = "sinfo -a --noheader"
    rows = parse_cmd(cmd)
    partitions = set()
    for row in rows:
        par = row.split()[0]
        if any([k in par for k in keywords]):
            partitions.add(par)
    return sorted(list(partitions))


@functools.lru_cache(maxsize=64, typed=True)
def occupancy_stats_for_node(node: str) -> dict:
    """Query SLURM for the occupancy of a given node.

    Args:
        (node): the name of the node to query

    Returns:
        a mapping between node names and occupancy stats.
    """
    cmd = f"scontrol show node {node}"
    rows = [x.strip() for x in parse_cmd(cmd)]
    keys = ("AllocTRES", "CfgTRES")
    metrics = {}
    for row in rows:
        for key in keys:
            if row.startswith(key):
                row = row.replace(f"{key}=", "")
                tokens = row.split(",")
                if tokens == [""]:
                    # SLURM sometimes omits information, so we alert the user to its
                    # its exclusion and report nothing for this node
                    print(f"Missing information for {node}: {key}, skipping....")
                    metrics[key] = {}
                else:
                    metrics[key] = {token.split("=", 1)[0]: token.split("=", 1)[1] 
                                   for token in tokens if "=" in token}
    occupancy = {}
    for metric, alloc_val in metrics["AllocTRES"].items():
        cfg_val = metrics["CfgTRES"][metric]
        if metric == "mem":
            # SLURM appears to sometimes misformat large numbers, producing summary strings
            # like 68G/257669M, rather than 68G/258G. The humanfriendly library provides
            # a more reliable number parser, and the humanize library provides a nice
            # formatter.
            alloc_val = format_size(hf.parse_size(alloc_val,binary=True))
            cfg_val = format_size(hf.parse_size(cfg_val,binary=True))
        occupancy[metric] = f"{alloc_val}/{cfg_val}"
    return occupancy

def format_size(size_in_bytes):
    """Format size in bytes to human readable string (G or T)."""
    size_in_gb = size_in_bytes / (1024 ** 3)  
    if size_in_gb < 1024: 
        return f"{int(size_in_gb)} G"
    
    size_in_tb = size_in_gb / 1024
    formatted_tb = f"{size_in_tb:.1f}".rstrip('0').rstrip('.')
    return f"{formatted_tb} T"

def lru_cache_time(seconds, maxsize=None):
    """
    Adds time aware caching to lru_cache.

    https://stackoverflow.com/questions/31771286/python-in-memory-cache-with-time-to-live
    """
    def wrapper(func):
        # Lazy function that makes sure the lru_cache() invalidate after X secs
        ttl_hash = lazy(lambda: round(time.time() / seconds), int)()
        
        @functools.lru_cache(maxsize)
        def time_aware(__ttl, *args, **kwargs):
            """
            Main wrapper, note that the first argument ttl is not passed down. 
            This is because no function should bother to know this that 
            this is here.
            """
            def wrapping(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapping(*args, **kwargs)
        return functools.update_wrapper(functools.partial(time_aware, ttl_hash), func)
    return wrapper


def _tres_occupancy(node: str, rows: list) -> dict:
    """Free/total cpu and mem for one `scontrol show node` record."""
    keys = ("AllocTRES", "CfgTRES")
    metrics = {}
    for row in rows:
        for key in keys:
            if row.startswith(key):
                row = row.replace(f"{key}=", "")
                tokens = row.split(",")
                if tokens == [""]:
                    # SLURM sometimes omits information, so we alert the user to its
                    # its exclusion and report nothing for this node
                    # print(f"Missing information for {node}: {key}, skipping....")
                    metrics[key] = {}
                else:
                    metrics[key] = {token.split("=", 1)[0]: token.split("=", 1)[1]
                                   for token in tokens if "=" in token}
    if "CfgTRES" not in metrics or "AllocTRES" not in metrics:
        print(f"Warning: node {node} reported no TRES totals, showing it as empty")
        return {"cpu": "0 / 0", "mem": "0 G / 0 G"}
    occupancy = {}
    for metric, cfg_val in metrics["CfgTRES"].items():
        try:
            alloc_val = metrics["AllocTRES"][metric]
        except KeyError:
            alloc_val = '0'
        if metric == "mem":
            # SLURM appears to sometimes misformat large numbers, producing summary strings
            # like 68G/257669M, rather than 68G/258G. The humanfriendly library provides
            # a more reliable number parser, and the humanize library provides a nice
            # formatter.
            avail_val = format_size(hf.parse_size(cfg_val,binary=True) - hf.parse_size(alloc_val,binary=True))
            cfg_val = format_size(hf.parse_size(cfg_val,binary=True))
            occupancy[metric] = f"{avail_val} / {cfg_val}"
        else:
            occupancy[metric] = f"{hf.parse_size(cfg_val)-hf.parse_size(alloc_val)} / {hf.parse_size(cfg_val)}"
    return occupancy


@lru_cache_time(seconds=10)
def _avail_stats_by_node() -> dict:
    """Availability for every node on the cluster, from a single scontrol call.

    Asking node by node costs one RPC each, and the resource panel walks about a
    hundred GPU nodes, so that alone was several seconds of every page refresh.
    One `scontrol show node` returns all of them in about the time a single-node
    query takes, so read the whole table once and let callers index into it.
    Records start at an unindented NodeName=; every other line is indented.
    """
    try:
        rows = parse_cmd("scontrol show node")
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Warning: could not read the node table ({exc}), showing every node as empty")
        return {}
    records, node = {}, None
    for row in rows:
        if row.startswith("NodeName="):
            node = row.split()[0].split("=", 1)[1]
            records[node] = []
        elif node is not None:
            records[node].append(row.strip())
        else:
            print(f"Warning: line before any NodeName= in scontrol output, ignoring it: {row.strip()!r}")
    return {name: _tres_occupancy(name, rows) for name, rows in records.items()}


def avail_stats_for_node(node: str) -> dict:
    """Query SLURM for the availability of a given node.

    Args:
        (node): the name of the node to query

    Returns:
        a mapping between node names and availability stats.
    """
    stats = _avail_stats_by_node().get(node)
    if stats is None:
        print(f"Warning: no scontrol record for node {node}, showing it as empty")
        return {"cpu": "0 / 0", "mem": "0 G / 0 G"}
    # The cache hands out one dict per node, so copy it rather than let a caller
    # edit what the next caller will read.
    return dict(stats)


@beartype
def parse_all_gpus(partition: Optional[str] = None,
                   default_gpus: int = 4,
                   default_gpu_name: str = "NONAME_GPU") -> dict:
    """Query SLURM for the number and types of GPUs under management.

    Args:
        partition: the partition/queue (or multiple, comma separated) of interest.
            By default None, which queries all available partitions.
        default_gpus: The number of GPUs estimated for nodes that have incomplete SLURM
            meta data.
        default_gpu_name: The name of the GPU for nodes that have incomplete SLURM meta
        data.

    Returns:
        a mapping between node names and a list of the GPUs that they have available.
    """
    cmd = "sinfo -o '%5000N|%5000G' --noheader"
    if partition:
        cmd += f" --partition={partition}"
    else:
        cmd += " -a"
    rows = parse_cmd(cmd)
    resources = defaultdict(list)

    # Debug the regular expression below at
    # https://regex101.com/r/RHYM8Z/3
    # Updated regex to support MIG GPU types like "1g.10gb" which contain dots
    p = re.compile(r'gpu:(?:([^:]*):)?(\d*)(?:\(\S*\))?\s*')

    for row in rows:
        node_str, resource_strs = row.split("|")
        for resource_str in resource_strs.split(","):
            if not resource_str.startswith("gpu"):
                continue
            match = p.search(resource_str)
            gpu_type = match.group(1) if match.group(1) is not None else default_gpu_name
            # if the number of GPUs is not specified, we assume it is `default_gpus`
            gpu_count = int(match.group(2)) if match.group(2) != "" else default_gpus
            node_names = parse_node_names(node_str)
            for name in node_names:
                resources[name].append({"type": gpu_type, "count": gpu_count})
    return resources


@beartype
def resource_by_type(resources: dict) -> dict:
    """Determine the cluster capacity by gpu type

    Args:
        resources: a summary of the cluster resources, organised by node name.

    Returns:
        resources: a summary of the cluster resources, organised by gpu type
    """
    by_type = defaultdict(list)
    for node, specs in resources.items():
        for spec in specs:
            by_type[spec["type"]].append({"node": node, "count": spec["count"]})
    return by_type


@beartype
def summary_by_type(resources: dict, tag: str):
    """Print out out a summary of cluster resources, organised by gpu type.

    Args:
        resources (dict): a summary of cluster resources, organised by node name.
        tag (str): a term that will be included in the printed summary.
    """
    by_type = resource_by_type(resources)
    total = sum(x["count"] for sublist in by_type.values() for x in sublist)
    agg_str = []
    for key, val in sorted(by_type.items(), key=lambda x: sum(y["count"] for y in x[1])):
        gpu_count = sum(x["count"] for x in val)
        agg_str.append(f"{gpu_count} {key} gpus")
    print(f"There are a total of {total} gpus [{tag}]")
    print("\n".join(agg_str))


@beartype
def summary(mode: str, resources: dict = None, states: dict = None):
    """Generate a printed summary of the cluster resources.

    Args:
        mode (str): the kind of resources to query (must be one of 'accessible', 'up').
        resources (dict :: None): a summary of cluster resources, organised by node name.
        states (dict[str: str] :: None): a mapping between node names and SLURM states.
    """
    if not resources:
        resources = parse_all_gpus()
    if not states:
        states = node_states()
    if mode == "accessible":
        res = {key: val for key, val in resources.items()
               if not is_inaccessible(states.get(key, "down"))}
    elif mode == "up":
        res = resources
    else:
        raise ValueError(f"Unknown mode: {mode}")
    summary_by_type(res, tag=mode)


@functools.lru_cache(maxsize=1)
def _get_slurm_version():
    return parse_cmd("sinfo -V", split=False).split(" ")[1]


@beartype
def gpu_usage(resources: dict, partition: Optional[str] = "gpu-a40,gpu-v100,gpu-a100-80,gpu-a100-40,gpu-a6000,gpu-b200,gpu-rtxpro6000,interactive-rtx3090,interactive-rtx2080,gpu-h200,gpu-mig,gpu-mig-a100,gpu-mig-rtxpro6000,dedicated") -> dict:
    """Build a data structure of the cluster resource usage, organised by user.

    Args:
        resources (dict :: None): a summary of cluster resources, organised by node name.

    Returns:
        (dict): a summary of resources organised by user (and also by node name).
    """
    slurm_version = _get_slurm_version()
    if slurm_version.startswith("17"):
       resource_flag = "gres"
    else:
       resource_flag = "tres-per-node"

    if int(slurm_version[0:2]) >= 21:
        gpu_identifier = 'gres/gpu'
    else:
        gpu_identifier = 'gpu'

    cmd = f"squeue -a -O {resource_flag}:100,nodelist:100,username:100,jobid:100,BatchFlag:10 --noheader"
    if partition:
        cmd += f" --partition={partition}"
    rows = parse_cmd(cmd)
    usage = defaultdict(dict)
    for row in rows:
        tokens = row.split()
        # ignore pending jobs
        if len(tokens) < 5 or not tokens[0].startswith(gpu_identifier):
            continue
        gpu_count_str, node_str, user, jobid, batch_flag = tokens
        gpu_count_tokens = gpu_count_str.split(":")
        if not gpu_count_tokens[-1].isdigit():
            gpu_count_tokens.append("1")
        num_gpus = int(gpu_count_tokens[-1])
        is_bash = batch_flag.strip() == "0"
        num_bash_gpus = num_gpus * is_bash
        node_names = parse_node_names(node_str)
        for node_name in node_names:
            # If a node still has jobs running but is draining, it will not be present
            # in the "available" resources, so we ignore it
            if node_name not in resources:
                continue
            node_gpu_types = [x["type"] for x in resources[node_name]]
            if (len(gpu_count_tokens) == 2) or (int(slurm_version[0:2]) >= 21):
                gpu_type = None
            elif len(gpu_count_tokens) == 3:
                gpu_type = gpu_count_tokens[1]
            if gpu_type is None:
                if len(node_gpu_types) != 1:
                    gpu_type = sorted(
                        resources[node_name],
                        key=lambda k: k['count'],
                        reverse=True
                    )[0]['type']
                    msg = (f"cannot determine node gpu type for {user} on {node_name}"
                           f" (guessing {gpu_type})")
                    print(f"WARNING >>> {msg}")
                else:
                    gpu_type = node_gpu_types[0]
            if gpu_type in usage[user]:
                usage[user][gpu_type][node_name]['n_gpu'] += num_gpus
                usage[user][gpu_type][node_name]['bash_gpu'] += num_bash_gpus

            else:
                usage[user][gpu_type] = defaultdict(lambda: {'n_gpu': 0, 'bash_gpu': 0})
                usage[user][gpu_type][node_name]['n_gpu'] += num_gpus
                usage[user][gpu_type][node_name]['bash_gpu'] += num_bash_gpus

    return usage


@beartype
def in_use(resources: dict = None, partition: Optional[str] = None):
    """Print a short summary of the resources that are currently used by each user.

    Args:
        resources: a summary of cluster resources, organised by node name.
    """
    if not resources:
        resources = parse_all_gpus()
    usage = gpu_usage(resources, partition=partition)
    aggregates = {}
    for user, subdict in usage.items():
        aggregates[user] = {}
        aggregates[user]['n_gpu'] = {key: sum([x['n_gpu'] for x in val.values()])
                                     for key, val in subdict.items()}
        aggregates[user]['bash_gpu'] = {key: sum([x['bash_gpu'] for x in val.values()])
                                        for key, val in subdict.items()}
    print("Usage by user:")
    for user, subdict in sorted(aggregates.items(),
                                key=lambda x: sum(x[1]['n_gpu'].values())):
        total = (f"total: {str(sum(subdict['n_gpu'].values())):2s} "
                 f"(interactive: {str(sum(subdict['bash_gpu'].values())):2s})")
        summary_str = ", ".join([f"{key}: {val}" for key, val in subdict['n_gpu'].items()])
        print(f"{user:10s} [{total}] {summary_str}")


@beartype
def available(
        resources: dict = None,
        states: dict = None,
        verbose: bool = False,
):
    """Print a short summary of resources available on the cluster.

    Args:
        resources: a summary of cluster resources, organised by node name.
        states: a mapping between node names and SLURM states.
        verbose: whether to output a more verbose summary of the cluster state.

    NOTES: Some systems allow users to share GPUs.  The logic below amounts to a
    conservative estimate of how many GPUs are available.  The algorithm is:

      For each user that requests a GPU on a node, we assume that a new GPU is allocated
      until all GPUs on the server are assigned.  If more GPUs than this are listed as
      allocated by squeue, we assume any further GPU usage occurs by sharing GPUs.
    """
    if not resources:
        resources = parse_all_gpus()
    if not states:
        states = node_states()
    res = {key: val for key, val in resources.items()
           if not is_inaccessible(states.get(key, "down"))}
    usage = gpu_usage(resources=res)
    for subdict in usage.values():
        for gpu_type, node_dicts in subdict.items():
            for node_name, user_gpu_count in node_dicts.items():
                resource_idx = [x["type"] for x in res[node_name]].index(gpu_type)
                count = res[node_name][resource_idx]["count"]
                count = max(count - user_gpu_count['n_gpu'], 0)
                res[node_name][resource_idx]["count"] = count
    by_type = resource_by_type(res)
    total = sum(x["count"] for sublist in by_type.values() for x in sublist)
    print(f"There are {total} gpus available:")
    for key, counts_for_gpu_type in by_type.items():
        gpu_count = sum(x["count"] for x in counts_for_gpu_type)
        tail = ""
        if verbose:
            summary_strs = []
            for x in counts_for_gpu_type:
                node, count = x["node"], x["count"]
                if count:
                    occupancy = occupancy_stats_for_node(node)
                    users = [user for user in usage if node in usage[user].get(key, [])]
                    details = [f"{key}: {val}" for key, val in sorted(occupancy.items())]
                    details = f"[{', '.join(details)}] [{','.join(users)}]"
                    summary_strs.append(f"\n -> {node}: {count} {key} {details}")
            tail = " ".join(summary_strs)
        print(f"{key}: {gpu_count} available {tail}")


def all_info(color: int, verbose: bool, partition: Optional[str] = None):
    divider, slurm_str = "---------------------------------", "SLURM"
    # Use only verified color names from `colored` library
    colors = ["red", "green", "yellow", "blue", "magenta", "cyan", "white", "black"]
    if color:
        divider = colored.fg(colors[7]) + divider + colored.attr("reset")
        slurm_str = colored.fg(colors[0]) + slurm_str + colored.attr("reset")
    print(divider)
    print(f"Under {slurm_str} management")
    print(divider)
    # Remaining function logic...
    resources = parse_all_gpus(partition=partition)
    states = node_states(partition=partition)
    for mode in ("up", "accessible"):
        summary(mode=mode, resources=resources, states=states)
        print(divider)
    in_use(resources, partition=partition)
    print(divider)
    available(resources=resources, states=states, verbose=verbose)
    print(divider)

def main():
    parser = argparse.ArgumentParser(description="slurm_gpus tool")
    parser.add_argument("--action", default="current",
                        choices=["current", "history", "daemon-start", "daemon-stop"],
                        help=("The function performed by slurm_gpustat: `current` will"
                              " provide a summary of current usage, 'history' will "
                              "provide statistics from historical data (provided that the"
                              "logging daemon has been running). 'daemon-start' and"
                              "'daemon-stop' will start and stop the daemon, resp."))
    parser.add_argument("-p", "--partition", default=None,
                        help=("the partition/queue (or multiple, comma separated) of"
                              " interest. By default set to all available partitions."))
    parser.add_argument("--log_path",
                        default=Path.home() / "data/daemons/logs/slurm_gpustat.log",
                        help="the location where daemon log files will be stored")
    parser.add_argument("--gpustat_pid",
                        default=Path.home() / "data/daemons/pids/slurm_gpustat.pid",
                        help="the location where the daemon PID file will be stored")
    parser.add_argument("--daemon_log_interval", type=int, default=43200,
                        help="time interval (secs) between stat logging (default 12 hrs)")
    parser.add_argument("--color", type=int, default=1, help="color output")
    parser.add_argument("--verbose", action="store_true",
                        help="provide a more detailed breakdown of resources")
    args = parser.parse_args()

    if args.action == "current":
        all_info(color=args.color, verbose=args.verbose, partition=args.partition)
    elif args.action == "history":
        data = GPUStatDaemon.deserialize_usage(args.log_path)
        historical_summary(data)
    elif args.action.startswith("daemon"):
        daemon = GPUStatDaemon(
            log_path=args.log_path,
            pidfile=args.gpustat_pid,
            log_interval=args.daemon_log_interval,
        )
        if args.action == "daemon-start":
            print("Starting daemon")
            daemon.start()
        elif args.action == "daemon-stop":
            print("Stopping daemon")
            daemon.stop()


if __name__ == "__main__":
    main()
