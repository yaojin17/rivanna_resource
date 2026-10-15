"""Credit to https://github.com/albanie/slurm_gpustat with minor modifications for web app.

A simple tool for summarising the current GPU usage on a slurm cluster.
"""

import os
import re
import time
import threading
import argparse
import functools
import subprocess
from typing import Optional
from collections import defaultdict

# Same reason as in app.py: the login node is close to its strict-overcommit
# limit, and a 40-thread OpenBLAS pool is both useless here and large enough to
# push allocations over it.  Nothing here uses numpy, but available_resources.py
# imports this module before pandas so that this runs before numpy is loaded.
for _blas_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_blas_var, "1")

import colored
import humanfriendly as hf
from beartype import beartype


# SLURM states which indicate that the node is not available for submitting jobs
INACCESSIBLE = {"drain*", "down*", "drng", "drain", "down"}
INACCESSIBLE_KEYWORDS = ("down", "drain", "drng", "resv", "inval", "plnd", "pow_up", "reboot")
INTERACTIVE_CMDS = {"bash", "zsh", "sh"}
# How hard parse_cmd tries before letting a SLURM query fail, see its docstring.
SLURM_RETRIES = 3
SLURM_RETRY_DELAY = 1.0
# The GPU partitions the monitor covers. squeue's --partition matches the partition
# a job was submitted to, not the node it runs on, so a partition left out here drops
# every GPU its jobs hold -- even `gpu` and `interactive`, whose GPU nodes all belong
# to one of the typed partitions as well.
MONITORED_PARTITIONS = "gpu,gpu-a100-40,gpu-a100-80,gpu-a40,gpu-a6000,gpu-v100,gpu-b200,gpu-rtxpro6000,gpu-mig,gpu-mig-rtxpro6000,interactive,interactive-rtx2080,interactive-rtx3090,dedicated"

# The accounts offered in the page's account picker; the first one is what the
# page starts on. Everything account-scoped -- the allocation, priority, queue
# and wait-estimate panels -- follows whichever one is picked, so only accounts
# we would really submit under belong here.  uva_cv_lab2 is the paid account and
# is off limits, and dac_cheng has had no allocation since early 2026.
LAB_ACCOUNTS = tuple(
    a.strip()
    for a in os.environ.get("RIVANNA_ACCOUNTS", "uva_cv_lab,cang-lab-in-silico").split(",")
    if a.strip()
)


def is_inaccessible(state: str) -> bool:
    state_lower = state.lower().rstrip("*~$#%")
    if state_lower in INACCESSIBLE:
        return True
    return any(kw in state_lower for kw in INACCESSIBLE_KEYWORDS)


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


def ttl_cache(seconds):
    """Cache a function's results per argument tuple for ``seconds``.

    Expired entries are dropped whenever a new result is stored, so the cache only
    ever holds what was asked for within the last ``seconds``.
    """
    def decorate(func):
        cache = {}
        guard = threading.Lock()

        @functools.wraps(func)
        def cached(*args):
            now = time.monotonic()
            with guard:
                hit = cache.get(args)
            if hit is not None and now - hit[0] < seconds:
                return hit[1]
            value = func(*args)
            with guard:
                for key in [k for k, (stamp, _) in cache.items() if now - stamp >= seconds]:
                    del cache[key]
                cache[args] = (now, value)
            return value
        return cached
    return decorate


def squeue_fields(args: str, fields) -> list:
    """Run ``squeue <args> -O`` over ``fields`` and return one dict per job.

    Given a width, squeue -O cuts a field to it without a word, and a node list cut
    inside its brackets cannot be parsed. So every field is asked for in full
    (width 0) and closed with ``|``. Only the last field may itself contain a
    ``|``, so free text such as a job name or pending reason goes last; a row
    that still does not split into ``fields`` is skipped with a warning.
    """
    fmt = ",".join(f"{name}:0|" for name in fields)
    entries = []
    for row in parse_cmd(f"squeue {args} -O '{fmt}'"):
        values = row.rstrip().removesuffix("|").split("|", len(fields) - 1)
        if len(values) != len(fields) or not row.rstrip().endswith("|"):
            print(f"Warning: squeue row does not split into {len(fields)} fields: "
                  f"{row[:120]!r}, skipping")
            continue
        entries.append({name: value.strip() for name, value in zip(fields, values)})
    return entries


# Every field any panel or the CLI reads off a running or pending job. A filtered
# squeue (-A, -p) takes about as long as a full one, so each state is read once in
# full and shared, and callers filter the rows themselves. reason is free text and so
# goes last, see squeue_fields.
JOB_FIELDS = (
    "jobid", "arrayjobid", "arraytaskid", "prioritylong", "partition", "account",
    "username", "state", "batchflag", "submittime", "starttime", "endtime", "timeused",
    "timelimit", "nodelist", "numnodes", "tres-per-node", "tres-per-job",
    "tres-per-task", "tres-alloc", "reason",
)


def squeue_jobs(state: str, partitions: Optional[str] = None) -> list:
    """Every job squeue -t ``state`` lists, as JOB_FIELDS dicts in squeue's order.

    ``partitions`` (comma separated) keeps the jobs ``squeue --partition`` would:
    those with any of them among the partitions they list.
    """
    jobs = squeue_fields(f"-a -h -t {state}", JOB_FIELDS)
    if partitions:
        wanted = set(partitions.split(","))
        jobs = [job for job in jobs if wanted & set(job["partition"].split(","))]
    return jobs


_GRES_GPU = re.compile(r"gpu:(?:([^:]*):)?(\d*)(?:\(\S*\))?")


def parse_gres_gpus(gres: str) -> list:
    """``(type, count)`` for each GPU entry of a sinfo or scontrol gres string.

    ``gpu:a100:8(S:0-1)`` gives ``[("a100", 8)]``. The type is None when the entry
    names none and the count None when it gives none; what that means is up to
    the caller. Entries that are not GPUs are ignored.
    """
    found = []
    for entry in gres.strip().split(","):
        if not entry.startswith("gpu"):
            continue
        match = _GRES_GPU.search(entry)
        if not match:
            print(f"Warning: could not parse gres entry {entry!r}, skipping it")
            continue
        found.append((match.group(1) or None, int(match.group(2)) if match.group(2) else None))
    return found


def tres_field(tres, name):
    """Value of ``name`` in a TRES string like ``cpu=8,mem=64G,gres/gpu=2``."""
    match = re.search(rf"(?:^|,){re.escape(name)}=([^,]+)", tres)
    return match.group(1) if match else None


def mem_to_mb(text):
    """``48G`` / ``4000000M`` / ``7812.50G`` / ``2T`` to megabytes."""
    match = re.fullmatch(r"([0-9.]+)([KMGT]?)", text.strip())
    if not match:
        return None
    value = float(match.group(1))
    scale = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[match.group(2)]
    return int(value * scale)


def alloc_gpu_count(tres):
    """GPUs in a TRES string, counting the typed form (``gres/gpu:a40=1``) too."""
    total = tres_field(tres, "gres/gpu")
    if total is not None and total.isdigit():
        return int(total)
    typed = re.findall(r"(?:^|,)gres/gpu:[^=,]+=(\d+)", tres)
    return sum(int(n) for n in typed)


def reserved_for_others():
    """Nodes held by an active reservation that none of LAB_ACCOUNTS may use.

    sinfo only reports a reserved node as ``resv`` while it is idle; once any job
    runs on it the state reads plain ``mix``, so without this a node like
    udc-ba07-38 (reservation dac_ssz, one CPU in use, eight idle A40s) looks
    wide open. Returns ``{node: reservation name}``.

    Access is judged by account only. A reservation granted to users or groups
    (maintenance windows held for root, say) is treated as closed, and
    ``Accounts=-name`` is SLURM's "everyone except" form.
    """
    locked = {}
    for row in parse_cmd("scontrol show reservation -o"):
        fields = dict(
            token.split("=", 1) for token in row.split() if "=" in token
        )
        name = fields.get("ReservationName", "?")
        if fields.get("State") != "ACTIVE":
            continue
        nodes = fields.get("Nodes", "(null)")
        if nodes in ("", "(null)"):
            print(f"Warning: active reservation {name} names no nodes, ignoring it")
            continue
        accounts = fields.get("Accounts", "(null)")
        listed = set() if accounts == "(null)" else set(accounts.lstrip("-").split(","))
        if accounts.startswith("-"):
            ours = not (listed & set(LAB_ACCOUNTS))
        else:
            ours = bool(listed & set(LAB_ACCOUNTS))
        if ours:
            continue
        for node in parse_node_names(nodes):
            # A node in two closed reservations just keeps the first name.
            locked.setdefault(node, name)
    return locked


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


def format_size(size_in_bytes):
    """Format size in bytes to human readable string (G or T)."""
    size_in_gb = size_in_bytes / (1024 ** 3)
    if size_in_gb < 1024:
        return f"{int(size_in_gb)} G"

    size_in_tb = size_in_gb / 1024
    formatted_tb = f"{size_in_tb:.1f}".rstrip('0').rstrip('.')
    return f"{formatted_tb} T"


@ttl_cache(10)
def node_records() -> dict:
    """``{node: {key: value}}`` for every node, from a single `scontrol show node`.

    Asking node by node costs one RPC each, and the resource panel walks about a
    hundred GPU nodes, so that alone was several seconds of every page refresh.
    One `scontrol show node` returns all of them in about the time a single-node
    query takes, so read the whole table once and let callers index into it.
    Records start at an unindented NodeName=; every other line is indented. Values
    are the raw strings up to the next space, which covers every key read here
    (OS= and Reason=, which hold spaces, keep only their first word).
    """
    try:
        rows = parse_cmd("scontrol show node")
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Warning: could not read the node table ({exc}), showing every node as empty")
        return {}
    records, record = {}, None
    for row in rows:
        if row.startswith("NodeName="):
            record = records.setdefault(row.split()[0].split("=", 1)[1], {})
        elif record is None:
            print(f"Warning: line before any NodeName= in scontrol output, ignoring it: {row.strip()!r}")
            continue
        for token in row.split():
            key, sep, value = token.partition("=")
            if sep:
                record.setdefault(key, value)
    return records


def _tres_dict(text):
    """``cpu=8,mem=64G`` -> ``{"cpu": "8", "mem": "64G"}``."""
    return dict(token.split("=", 1) for token in text.split(",") if "=" in token)


def avail_stats_for_node(node: str) -> dict:
    """Free and total CPUs and memory of one node.

    Returns ``{"cpu": (free, total), "mem": (free bytes, total bytes)}``. SLURM
    sometimes writes large memory sizes oddly (68G next to 257669M), so they go
    through humanfriendly's parser rather than being compared as text.
    """
    record = node_records().get(node)
    if record is None:
        print(f"Warning: no scontrol record for node {node}, showing it as empty")
        return {"cpu": (0, 0), "mem": (0, 0)}
    if "CfgTRES" not in record or "AllocTRES" not in record:
        print(f"Warning: node {node} reported no TRES totals, showing it as empty")
        return {"cpu": (0, 0), "mem": (0, 0)}
    cfg, alloc = _tres_dict(record["CfgTRES"]), _tres_dict(record["AllocTRES"])
    stats = {}
    for metric in ("cpu", "mem"):
        total = hf.parse_size(cfg[metric], binary=metric == "mem")
        used = hf.parse_size(alloc.get(metric, "0"), binary=metric == "mem")
        stats[metric] = (total - used, total)
    return stats


def occupancy_stats_for_node(node: str) -> dict:
    """Allocated/total of every TRES SLURM tracks on ``node``, as display strings."""
    record = node_records().get(node, {})
    cfg = _tres_dict(record.get("CfgTRES", ""))
    occupancy = {}
    for metric, alloc_val in _tres_dict(record.get("AllocTRES", "")).items():
        cfg_val = cfg[metric]
        if metric == "mem":
            alloc_val = format_size(hf.parse_size(alloc_val, binary=True))
            cfg_val = format_size(hf.parse_size(cfg_val, binary=True))
        occupancy[metric] = f"{alloc_val}/{cfg_val}"
    return occupancy


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
    cmd = "sinfo -o '%N|%G' --noheader"
    if partition:
        cmd += f" --partition={partition}"
    else:
        cmd += " -a"
    rows = parse_cmd(cmd)
    resources = defaultdict(list)

    for row in rows:
        node_str, resource_strs = row.split("|")
        for gpu_type, gpu_count in parse_gres_gpus(resource_strs):
            gpu_type = gpu_type or default_gpu_name
            # if the number of GPUs is not specified, we assume it is `default_gpus`
            gpu_count = default_gpus if gpu_count is None else gpu_count
            for name in parse_node_names(node_str):
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
def _gpus_per_node_from_alloc(num_nodes: str, tres_alloc: str, jobid: str) -> int:
    """GPUs per node for a job whose tres-per-node names none.

    tres-alloc's ``gres/gpu=N`` is the job's total over all its nodes.
    """
    total = alloc_gpu_count(tres_alloc)
    nodes = int(num_nodes)
    if total % nodes:
        print(f"Warning: job {jobid} holds {total} GPUs over {nodes} nodes, "
              f"counting {total // nodes} per node")
    return total // nodes


@beartype
def gpu_usage(resources: dict, partition: Optional[str] = MONITORED_PARTITIONS) -> dict:
    """Build a data structure of the cluster resource usage, organised by user.

    Args:
        resources (dict :: None): a summary of cluster resources, organised by node name.

    Returns:
        (dict): a summary of resources organised by user (and also by node name).
    """
    usage = defaultdict(dict)
    # Tallied and reported once below rather than per job: these skips are routine
    # (CPU-only jobs, drained nodes) and a refresh walks hundreds of jobs.
    no_gpu_jobs = 0
    off_inventory = defaultdict(int)  # node -> GPUs held there that go uncounted
    # RUNNING only: a COMPLETING job still lists its nodes, but SLURM has
    # already given its GPUs back to the node and to the user's QOS limit.
    for job in squeue_jobs("RUNNING", partition):
        user, jobid = job["username"], job["jobid"]
        if job["tres-per-node"].startswith("gres/gpu"):
            # gres/gpu:N or gres/gpu:<type>:N; a bare gres/gpu means one
            count = job["tres-per-node"].split(":")[-1]
            num_gpus = int(count) if count.isdigit() else 1
        else:
            # A job that asks with --gpus or --gpus-per-task has no tres-per-node; its
            # GPUs only show up in its allocation.
            num_gpus = _gpus_per_node_from_alloc(job["numnodes"], job["tres-alloc"], jobid)
            # A CPU-only job (e.g. on the dedicated nodes) holds no GPU to count
            if not num_gpus:
                no_gpu_jobs += 1
                continue
        num_bash_gpus = num_gpus * (job["batchflag"] == "0")
        for node_name in parse_node_names(job["nodelist"]):
            # If a node still has jobs running but is draining, it will not be present
            # in the "available" resources, so we ignore it
            if node_name not in resources:
                off_inventory[node_name] += num_gpus
                continue
            # The type comes from the node: a request need not name one.
            node_gpus = resources[node_name]
            gpu_type = max(node_gpus, key=lambda k: k["count"])["type"]
            if len(node_gpus) != 1:
                print(f"WARNING >>> cannot determine node gpu type for {user} on {node_name}"
                      f" (guessing {gpu_type})")
            counts = usage[user].setdefault(
                gpu_type, defaultdict(lambda: {'n_gpu': 0, 'bash_gpu': 0}))[node_name]
            counts['n_gpu'] += num_gpus
            counts['bash_gpu'] += num_bash_gpus

    if no_gpu_jobs:
        print(f"Note: gpu_usage skipped {no_gpu_jobs} running jobs that hold no GPU")
    if off_inventory:
        nodes = sorted(off_inventory)
        shown = ", ".join(nodes[:5]) + (f" and {len(nodes) - 5} more" if len(nodes) > 5 else "")
        print(f"Warning: gpu_usage left out {sum(off_inventory.values())} GPUs held on "
              f"{len(nodes)} nodes outside the inventory it was given ({shown})")
    return usage


def aggregate_usage(usage: dict) -> dict:
    """Fold gpu_usage's per-node counts into per-user totals by GPU type.

    Returns ``{user: {"n_gpu": {type: n}, "bash_gpu": {type: n}}}``.
    """
    return {
        user: {
            field: {gpu_type: sum(x[field] for x in nodes.values())
                    for gpu_type, nodes in by_type.items()}
            for field in ("n_gpu", "bash_gpu")
        }
        for user, by_type in usage.items()
    }


def subtract_usage(resources: dict, usage: dict) -> None:
    """Take the GPUs in ``usage`` off the per-node counts in ``resources``, in place.

    A node never drops below zero: GPUs held past its count are shared ones.
    """
    for subdict in usage.values():
        for gpu_type, node_dicts in subdict.items():
            for node_name, user_gpu_count in node_dicts.items():
                resource_idx = [x["type"] for x in resources[node_name]].index(gpu_type)
                count = resources[node_name][resource_idx]["count"]
                resources[node_name][resource_idx]["count"] = max(count - user_gpu_count["n_gpu"], 0)


@beartype
def in_use(resources: dict = None, partition: Optional[str] = None):
    """Print a short summary of the resources that are currently used by each user.

    Args:
        resources: a summary of cluster resources, organised by node name.
    """
    if not resources:
        resources = parse_all_gpus()
    aggregates = aggregate_usage(gpu_usage(resources, partition=partition))
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
    subtract_usage(res, usage)
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
    parser.add_argument("-p", "--partition", default=None,
                        help=("the partition/queue (or multiple, comma separated) of"
                              " interest. By default set to all available partitions."))
    parser.add_argument("--color", type=int, default=1, help="color output")
    parser.add_argument("--verbose", action="store_true",
                        help="provide a more detailed breakdown of resources")
    args = parser.parse_args()

    all_info(color=args.color, verbose=args.verbose, partition=args.partition)


if __name__ == "__main__":
    main()
