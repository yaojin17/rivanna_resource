import os

# The login node runs with strict overcommit (vm.overcommit_memory=2) and sits
# close to its commit limit, so the per-core buffers OpenBLAS reserves on one
# machine with 40 cores are enough to make allocations fail process-wide.  This
# app only does elementwise integer array work, no BLAS, so pin the pools to a
# single thread; it must happen before numpy is imported to take effect.
for _blas_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_blas_var, "1")

import re
from datetime import datetime, timedelta
import argparse
import functools
import json
import pwd
import bisect
import copy
import time
import numpy as np
import pytz
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from html import escape
from subprocess import (
    PIPE,
    STDOUT,
    CalledProcessError,
    TimeoutExpired,
    check_output,
    run,
)
from flask import Flask, Response, render_template_string, request
from slurm_gpustat import (
    resource_by_type,
    parse_all_gpus,
    gpu_usage,
    node_states,
    is_inaccessible,
    parse_cmd,
    avail_stats_for_node,
    parse_node_names,
    get_gpu_partitions,
)
# import gradio as gr  # Not used in this file

# from https://developer.nvidia.com/cuda-gpus
# sort gpu by computing power
# if fail to display other GPU types, add items in the following dictionaries.
CAPABILITY = {
    "1g.10gb": 8.0,
    "1g.24gb": 8.0,
    "b200": 10.0,
    "rtx_pro_6000": 12.0,
    "h200": 8.9,
    "a4500": 8.0,
    "a100": 8.0,
    "a6000": 8.6,
    "a40": 8.6,
    "a30": 8.0,
    "a10": 8.6,
    "a16": 8.6,
    "rtx_3090": 8.6,
    "rtx_2080": 7.5,
    "v100": 7.0,
    "gv100gl": 7.0,
    "v100s": 7.0,
    "p40": 6.1,
    "m40": 5.2,
    "rtx6k": 7.5,
    "rtx8k": 7.5,
}


def _type_rank(gpu_type):
    """Sort key putting the most capable GPU first, ties broken by name.

    Several types share a capability number (a40, a6000 and rtx_3090 are all
    8.6), and the lists being sorted are built from sets, whose order follows
    string hashing and so is randomised per process. Without the name the tied
    types came out in a different order every time the app was restarted.
    """
    return (-CAPABILITY.get(gpu_type, 10.0), gpu_type)

GMEM = {
    "mig": "[11g]",
    "1g.10gb": "[10g]",
    "1g.24gb": "[24g]",
    "b200": "[192g]",
    "rtx_pro_6000": "[96g]",
    "h200": "[141g]",
    "a4500": "[20g]",
    "a100": "[40/80g]",
    "a6000": "[48g]",
    "a40": "[48g]",
    "a30": "[24g]",
    "rtx_3090": "[24g]",
    "rtx_2080": "[11g]",
    "v100": "[16g]",
    "gv100gl": "[32g]",
    "v100s": "[32g]",
    "p40": "[24g]",
    "m40": "[12/24g]",
    "rtx6k": "[24g]",
    "rtx8k": "[48g]",
}
OLD_GPU_TYPES = ["p40", "m40"]


def get_resource_bar(avail, total, text="", long=False):
    """Create a long/short progress bar with text overlaid. Formatting handled in css."""

    if long:
        long_str = " class=long"
    else:
        long_str = ""
    if total == 0:
        total = 1  # avoid ZeroDivisionError Error
    bar = (
        f'<div class="progress" data-text="{text}">'
        f'<progress{long_str} max="100" value="{avail/total*100}"></progress></div>'
    )
    return bar


def str_to_int(text):
    """Convert string with unit to int in GB. e.g. "30 G" --> 30, "1.4 T" --> 1433."""
    text = text.strip()
    parts = text.split()
    if len(parts) != 2:
        return 0
    try:
        number = float(parts[0])
    except ValueError:
        print(f"Warning: str_to_int could not parse number from '{text}'")
        return 0
    unit = parts[1].upper().rstrip("B")
    if unit == 'T':
        return int(number * 1024)
    elif unit == 'G':
        return int(number)
    elif unit == 'M':
        return int(number / 1024)
    else:
        return 0


@functools.lru_cache(maxsize=None)
def display_name(user):
    """Return "Real Name (computing_id)" using the passwd GECOS field."""
    try:
        gecos = pwd.getpwnam(user).pw_gecos
    except KeyError:
        print(f"Warning: no passwd entry for user {user!r}, showing raw id")
        return user
    name = gecos.split(",")[0].replace("_", " ").strip()
    if not name:
        print(f"Warning: empty GECOS name for user {user!r}, showing raw id")
        return user
    return f"{name} ({user})"


def _leaderboard_row(user, subdict, name_width, sum_by_gmem, owner):
    """One leaderboard line; the row of the account running this app is marked."""
    total = f"total={str(sum(subdict['n_gpu'].values())):2s}"
    user_summary = [
        f"{key}={val}"
        for key, val in sorted(
            subdict["n_gpu"].items(),
            key=lambda x: _type_rank(x[0]),
        )
    ]
    summary_str = "".join([f"{i:<16s}" for i in user_summary])
    num_new_gpus = [
        val for key, val in subdict["n_gpu"].items() if key not in OLD_GPU_TYPES
    ]
    for gm in sum_by_gmem:
        total += f"|{gm}g={str(sum([val for key, val in subdict['n_gpu'].items() if key in GMEM and GMEM[key] == f'[{gm}g]'])):2s}"
    total += f"|newer={str(sum(num_new_gpus)):2s}"
    total += f"|bash={str(sum(subdict['bash_gpu'].values())):2s}"
    line = escape(f"{display_name(user):{name_width}s}[{total}]    {summary_str}")
    if user == owner:
        return f'<span class="leaderboard-me">&gt; {line}</span>\n'
    return f"  {line}\n"


def parse_leaderboard(sum_by_gmem=[48]):
    """Request sinfo, parse the leaderboard in string."""

    resources = parse_all_gpus()
    usage = gpu_usage(
        resources=resources,
    )  # partition='gpu'
    aggregates = {}
    for user, subdict in usage.items():
        aggregates[user] = {}
        aggregates[user]["n_gpu"] = {
            key: sum([x["n_gpu"] for x in val.values()]) for key, val in subdict.items()
        }
        aggregates[user]["bash_gpu"] = {
            key: sum([x["bash_gpu"] for x in val.values()])
            for key, val in subdict.items()
        }
    out = ""
    owner = _app_owner()
    name_width = max((len(display_name(u)) for u in aggregates), default=12) + 2
    for user, subdict in sorted(
        aggregates.items(), key=lambda x: sum(x[1]["n_gpu"].values()), reverse=True
    ):
        out += _leaderboard_row(user, subdict, name_width, sum_by_gmem, owner)
    return out


def parse_leaderboard_by_partition(sum_by_gmem=[48]):
    """Request sinfo, parse the leaderboard in string."""
    resources = parse_all_gpus()
    gpu_partitions = get_gpu_partitions()

    owner = _app_owner()
    out = "=" * 64 + "\n"
    for i, part in enumerate(gpu_partitions):
        usage = gpu_usage(resources=resources, partition=part)  # partition='gpu'
        aggregates = {}
        for user, subdict in usage.items():
            aggregates[user] = {}
            aggregates[user]["n_gpu"] = {
                key: sum([x["n_gpu"] for x in val.values()])
                for key, val in subdict.items()
            }
            aggregates[user]["bash_gpu"] = {
                key: sum([x["bash_gpu"] for x in val.values()])
                for key, val in subdict.items()
            }
        if i != 0:
            out += "-" * 64 + "\n"
        out += f"PARTITION: {part}\n"
        name_width = max((len(display_name(u)) for u in aggregates), default=12) + 2
        for user, subdict in sorted(
            aggregates.items(), key=lambda x: sum(x[1]["n_gpu"].values()), reverse=True
        ):
            out += _leaderboard_row(user, subdict, name_width, sum_by_gmem, owner)
    out += "=" * 64 + "\n"
    return out


def cpu_usage(resources, partition="compute"):
    """Build a data structure of the CPU resource usage, organised by user.

    Args:
        resources (dict :: None): a summary of cluster resources, organised by node name.

    Returns:
        (dict): a summary of resources organised by user (and also by node name).
    """
    cmd = "squeue -a -O NumNodes:100,nodelist:100,username:100,jobid:100 --noheader"
    if partition:
        cmd += f" --partition={partition}"
    rows = parse_cmd(cmd)
    usage = defaultdict(dict)
    for row in rows:
        tokens = row.split()
        # ignore pending jobs
        if len(tokens) < 4:
            continue
        cpu_count_str, node_str, user, jobid = tokens
        num_cpus = int(cpu_count_str.strip())
        node_names = parse_node_names(node_str)
        for node_name in node_names:
            # If a node still has jobs running but is draining, it will not be present
            # in the "available" resources, so we ignore it
            if node_name not in resources:
                continue
            cpu_type = resources[node_name]["type"]

            if cpu_type in usage[user]:
                usage[user][cpu_type][node_name]["n_cpu"] += num_cpus
            else:
                usage[user][cpu_type] = defaultdict(
                    lambda: {
                        "n_cpu": 0,
                    }
                )
                usage[user][cpu_type][node_name]["n_cpu"] += num_cpus
    return usage


def parse_cpu_usage_to_table(partition="compute", show_bar=True):
    """Request sinfo for cnode, parse the output to a html table."""

    node_str = parse_cmd(f"sinfo -o '%1000N' --noheader --partition={partition}")
    assert isinstance(node_str, list) and len(node_str) == 1
    node_names = parse_node_names(node_str[0].strip())
    resources = {k: {"type": k[0:6] + "xx", "count": 1} for k in node_names}
    states = node_states(partition=partition)
    res = {
        key: val
        for key, val in resources.items()
        if not is_inaccessible(states.get(key, "down"))
    }
    res_total = copy.deepcopy(res)
    usage = cpu_usage(resources=res, partition=partition)

    for subdict in usage.values():
        for cpu_type, node_dicts in subdict.items():
            for node_name, user_cpu_count in node_dicts.items():
                count = res[node_name]["count"]
                count = max(count - user_cpu_count["n_cpu"], 0)
                res[node_name]["count"] = count

    res_total_by_type = defaultdict(list)
    for node, spec in res_total.items():
        res_total_by_type[spec["type"]].append({"node": node, "count": spec["count"]})

    res_usage_by_type = defaultdict(list)
    for node, spec in res.items():
        res_usage_by_type[spec["type"]].append({"node": node, "count": spec["count"]})

    table_html = []
    total_cpu_count = 0
    avail_cpu_count = 0

    # sort cpus from new to old
    type_list = sorted(list(res_total_by_type.keys()), reverse=True)

    # writing the html table
    for cpu_type in type_list:
        node_dicts = res_total_by_type[cpu_type]
        node_names = sorted([i["node"] for i in node_dicts])

        node_summaries = []
        num_col = []

        for node in node_names:
            node_name = f"<td>{node}</td>"

            users = [user for user in usage if node in usage[user].get(cpu_type, [])]
            if len(users):
                users = f"<td>user: {','.join(users)}</td>"
            else:
                users = f"<td>&nbsp</td>"

            detail_dict = avail_stats_for_node(node)
            detail_dict = {k: v for k, v in detail_dict.items() if k in ["cpu", "mem"]}

            if show_bar:
                c_stat = detail_dict["cpu"].split("/")
                c_stat = [int(i.strip()) for i in c_stat]
                cpu_bar = get_resource_bar(*c_stat, text=detail_dict["cpu"])

                m_stat = detail_dict["mem"].split("/")
                m_stat = [str_to_int(i.strip()) for i in m_stat]
                mem_bar = get_resource_bar(*m_stat, text=detail_dict["mem"], long=True)

                total_cpu_count += c_stat[-1]
                avail_cpu_count += c_stat[-1] - c_stat[0]
            else:
                cpu_bar = detail_dict["cpu"]
                mem_bar = detail_dict["mem"]
            cpu_stat = f"<td>cpu: {cpu_bar}</td>"
            mem_stat = f"<td>mem: {mem_bar}</td>"

            node_summary = (
                f"<tr><td>&nbsp</td>{node_name}{cpu_stat}{mem_stat}{users}</tr>"
            )
            node_summaries.append(node_summary)
            num_col.append(5)

        type_summary = (
            f'<tr><td colspan="{max(num_col)}"><b>' f"{cpu_type}: </b></td></tr>"
        )
        table_html.append(type_summary)
        table_html.extend(node_summaries)

    if show_bar:
        total_bar = get_resource_bar(
            avail_cpu_count,
            total_cpu_count,
            text=f"{avail_cpu_count} / {total_cpu_count}",
        )
        total_summary = (
            f'<tr><td colspan="{max(num_col)}"><h3>'
            f"Summary: {total_bar} cpus available</h3></td></tr>"
        )
    else:
        total_bar = f"{avail_cpu_count}/{total_cpu_count}"
        total_summary = ""

    table_html = f"<table>{total_summary}{''.join(table_html)}</table>"

    return table_html


def _state_badge(state):
    """Return an HTML badge for the node state."""
    s = state.lower()
    if any(kw in s for kw in ("down", "drain", "drng", "inval")):
        cls = "state-down"
    elif any(kw in s for kw in ("resv", "plnd", "plan", "reboot", "pow_up")):
        cls = "state-resv"
    elif "mix" in s or "alloc" in s:
        cls = "state-busy"
    else:
        cls = "state-idle"
    return f'<span class="state-badge {cls}">{state}</span>'


MONITORED_PARTITIONS = "gpu,gpu-a100-40,gpu-a100-80,gpu-a40,gpu-a6000,gpu-v100,gpu-b200,gpu-rtxpro6000,gpu-mig,gpu-mig-a100,gpu-mig-rtxpro6000,interactive,interactive-rtx2080,interactive-rtx3090,dedicated"

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


def selected_account(args):
    """The account a request asked for, checked against the picker's list.

    A whitelist rather than a pattern match: these names are interpolated into
    the SLURM commands below, and an account nobody is a member of would only
    produce empty panels with no hint as to why.
    """
    wanted = args.get("account", "").strip()
    if not wanted:
        return LAB_ACCOUNTS[0]
    if wanted not in LAB_ACCOUNTS:
        print(f"Warning: unknown account {wanted!r} requested, using {LAB_ACCOUNTS[0]}")
        return LAB_ACCOUNTS[0]
    return wanted


def _reserved_for_others():
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

def _is_mig_type(gpu_type):
    """MIG slice types are named after their memory share, e.g. "1g.10gb"."""
    return gpu_type[0].isdigit() if gpu_type else False


def _gpu_pools():
    """Query SLURM once for the current GPU inventory of the monitored partitions.

    Returns a dict with the raw per-node data needed by the callers plus three
    views organised by gpu type: every GPU on a reachable node (`total_by_type`),
    the ones not currently allocated (`free_by_type`) and the ones stranded on
    unreachable nodes (`down_by_type`). The first two are also returned per node
    (`total_by_node`, `free_by_node`) for callers that need a per-partition slice.
    """
    resources = parse_all_gpus(partition=MONITORED_PARTITIONS)
    states = node_states(partition=MONITORED_PARTITIONS)
    reserved = {
        node: name for node, name in _reserved_for_others().items() if node in resources
    }
    # Reserved for somebody else is as good as down for us, whatever sinfo says.
    for node in reserved:
        states[node] = "resv"

    # Separate accessible vs inaccessible nodes
    res_accessible = {
        key: val
        for key, val in resources.items()
        if not is_inaccessible(states.get(key, "down"))
    }
    res_down = {
        key: val
        for key, val in resources.items()
        if is_inaccessible(states.get(key, "down"))
    }

    res_total = copy.deepcopy(res_accessible)
    usage = gpu_usage(resources=res_accessible)

    for subdict in usage.values():
        for gpu_type, node_dicts in subdict.items():
            for node_name, user_gpu_count in node_dicts.items():
                resource_idx = [x["type"] for x in res_accessible[node_name]].index(gpu_type)
                count = res_accessible[node_name][resource_idx]["count"]
                count = max(count - user_gpu_count["n_gpu"], 0)
                res_accessible[node_name][resource_idx]["count"] = count

    return {
        "states": states,
        "reserved": reserved,
        "usage": usage,
        "down_nodes": res_down,
        "total_by_node": res_total,
        "free_by_node": res_accessible,
        "total_by_type": resource_by_type(res_total),
        "free_by_type": resource_by_type(res_accessible),
        "down_by_type": resource_by_type(res_down),
    }


def parse_usage_to_table(show_bar=True):
    """Request sinfo, parse the output to a html table."""

    pools = _gpu_pools()
    states = pools["states"]
    usage = pools["usage"]
    res_down = pools["down_nodes"]
    reserved = pools["reserved"]
    res_total_by_type = pools["total_by_type"]
    res_usage_by_type = pools["free_by_type"]
    res_down_by_type = pools["down_by_type"]

    all_gpu_types = set(res_total_by_type.keys()) | set(res_down_by_type.keys())

    type_list = sorted(all_gpu_types, key=_type_rank)
    gpu_type_list = [t for t in type_list if not _is_mig_type(t)]
    mig_type_list = [t for t in type_list if _is_mig_type(t)]

    def _down_note(entries):
        """"(8 offline, 8 reserved)" for down-bucket entries, split by cause."""
        held = sum(x["count"] for x in entries if x["node"] in reserved)
        offline = sum(x["count"] for x in entries) - held
        parts = [f"{offline} offline"] * bool(offline) + [f"{held} reserved"] * bool(held)
        return f' <span class="down-note">({", ".join(parts)})</span>' if parts else ""

    def _render_type_rows(type_list_to_render):
        rows_html = []
        total_count = 0
        avail_count = 0
        num_col = [7]
        for gpu_type in type_list_to_render:
            node_dicts = res_total_by_type.get(gpu_type, [])
            down_node_dicts = res_down_by_type.get(gpu_type, [])
            accessible_node_names = sorted([i["node"] for i in node_dicts])
            down_node_names = sorted([i["node"] for i in down_node_dicts])
            gpu_count_total = {i["node"]: i["count"] for i in node_dicts}
            gpu_count_avail = {i["node"]: i["count"] for i in res_usage_by_type.get(gpu_type, [])}

            node_summaries = []
            down_summaries = []

            for node in accessible_node_names:
                node_state = states.get(node, "unknown")
                state_html = f"<td>{_state_badge(node_state)}</td>"
                node_name_td = f"<td>{node}</td>"
                if show_bar:
                    gpu_bar = get_resource_bar(
                        gpu_count_avail.get(node, 0),
                        gpu_count_total[node],
                        text=f"{gpu_count_avail.get(node, 0)} / {gpu_count_total[node]}",
                    )
                else:
                    gpu_bar = f"{gpu_count_avail.get(node, 0)}/{gpu_count_total[node]}"
                gpu_stat = f"<td>gpu: {gpu_bar}</td>"

                users_list = [user for user in usage if node in usage[user].get(gpu_type, [])]
                if len(users_list):
                    users_td = f"<td>user: {','.join(users_list)}</td>"
                else:
                    users_td = f"<td>&nbsp</td>"

                detail_dict = avail_stats_for_node(node)
                detail_dict = {k: v for k, v in detail_dict.items() if k in ["cpu", "mem"]}

                if show_bar:
                    c_stat = detail_dict["cpu"].split("/")
                    c_stat = [int(i.strip()) for i in c_stat]
                    cpu_bar = get_resource_bar(*c_stat, text=detail_dict["cpu"])

                    m_stat = detail_dict["mem"].split("/")
                    m_stat = [str_to_int(i.strip()) for i in m_stat]
                    mem_bar = get_resource_bar(*m_stat, text=detail_dict["mem"], long=True)
                else:
                    cpu_bar = detail_dict["cpu"]
                    mem_bar = detail_dict["mem"]
                cpu_stat = f"<td>cpu: {cpu_bar}</td>"
                mem_stat = f"<td>mem: {mem_bar}</td>"

                node_summaries.append(f"<tr><td>&nbsp</td>{node_name_td}{state_html}{gpu_stat}{cpu_stat}{mem_stat}{users_td}</tr>")

            for node in down_node_names:
                node_state = states.get(node, "unknown")
                state_html = f"<td>{_state_badge(node_state)}</td>"
                if node in reserved:
                    state_html = (
                        f"<td>{_state_badge(node_state)} "
                        f'<span class="queue-muted">{reserved[node]}</span></td>'
                    )
                node_name_td = f"<td>{node}</td>"
                node_gpu_count = sum(x["count"] for x in res_down[node])
                down_summaries.append(
                    f'<tr class="row-down"><td>&nbsp</td>{node_name_td}{state_html}'
                    f'<td class="dimmed">gpu: 0 / {node_gpu_count}</td>'
                    f'<td class="dimmed">-</td><td class="dimmed">-</td><td>&nbsp</td></tr>'
                )

            type_total = sum(gpu_count_total.values())
            type_avail = sum(gpu_count_avail.values())
            if show_bar:
                type_bar = get_resource_bar(type_avail, type_total, text=f"{type_avail} / {type_total}")
            else:
                type_bar = f"{type_avail}/{type_total}"

            down_note = _down_note(down_node_dicts)

            rows_html.append(
                f'<tr><td colspan="7"><b>{gpu_type} {GMEM.get(gpu_type, "")}: {type_bar} gpus available{down_note}</b></td></tr>'
            )
            rows_html.extend(node_summaries)
            rows_html.extend(down_summaries)
            total_count += type_total
            avail_count += type_avail

        down_entries = [x for t in type_list_to_render for x in res_down_by_type.get(t, [])]
        return "".join(rows_html), avail_count, total_count, down_entries

    gpu_rows, gpu_avail, gpu_total, gpu_down = _render_type_rows(gpu_type_list)
    mig_rows, mig_avail, mig_total, mig_down = _render_type_rows(mig_type_list)

    if show_bar:
        total_bar = get_resource_bar(gpu_avail, gpu_total, text=f"{gpu_avail} / {gpu_total}")
    else:
        total_bar = f"{gpu_avail}/{gpu_total}"
    down_note = _down_note(gpu_down)
    total_summary = f'<tr><td colspan="7"><h3>Summary: {total_bar} gpus available{down_note}</h3></td></tr>'

    table_html = f"<table>{total_summary}{gpu_rows}</table>"

    if mig_type_list:
        if show_bar:
            mig_bar = get_resource_bar(mig_avail, mig_total, text=f"{mig_avail} / {mig_total}")
        else:
            mig_bar = f"{mig_avail}/{mig_total}"
        mig_down_note = _down_note(mig_down)
        table_html += (
            f'<details class="mig-details"><summary>MIG: {mig_bar} slices available{mig_down_note}</summary>'
            f"<table>{mig_rows}</table></details>"
        )

    return table_html


def _count_array_tasks(jobid_str):
    """Count the number of tasks in a job array ID like '123_[0-5,8,10-12]'."""
    if "_[" not in jobid_str:
        return 1
    if "]" not in jobid_str:
        return 1
    bracket = jobid_str[jobid_str.index("[") + 1 : jobid_str.index("]")]
    bracket = bracket.split("%")[0]
    count = 0
    for part in bracket.split(","):
        if "-" in part:
            tokens = part.split("-", 1)
            if tokens[0].isdigit() and tokens[1].isdigit():
                count += int(tokens[1]) - int(tokens[0]) + 1
            else:
                count += 1
        else:
            count += 1
    return count


def _is_gpu_partition(partition_name):
    """Check if a partition name is GPU-related."""
    keywords = ("gpu", "interactive", "dedicated")
    return any(kw in partition_name.lower() for kw in keywords)


# Pending reasons meaning the job is genuinely queued for GPUs right now.
WAITING_REASONS = (
    "Priority",
    "Resources",
    "None",
    "ReqNodeNotAvail",
    "Nodes required for job are DOWN",
)
# Pending reasons meaning the job is real demand, but capped by a user/account/array
# limit: it cannot start until the same submitter's other jobs finish.
THROTTLED_REASON_KEYS = (
    "QOSMax",
    "QOSGrp",
    "QOSResourceLimit",
    "AssocMax",
    "AssocGrp",
    "JobArrayTaskLimit",
    "MaxJobsPer",
)
# Pending reasons meaning the job is not waiting on GPUs at all.
BLOCKED_REASON_KEYS = (
    "Dependency",
    "BeginTime",
    "JobHeld",
    "BadConstraints",
    "InvalidAccount",
    "InvalidQOS",
    "PartitionDown",
    "PartitionConfig",
    "PartitionTimeLimit",
    "PartitionNodeLimit",
    "AccountNotAllowed",
    "NodeDown",
    "Reservation",
    "Cleaning",
    "Licenses",
    "requeued held",
    "launch failed",
)

QUEUE_CATEGORIES = ("waiting", "throttled", "blocked")


def _classify_pending_reason(reason):
    """Bucket a squeue pending reason into waiting / throttled / blocked."""
    reason = reason.strip()
    if reason.startswith(WAITING_REASONS):
        return "waiting"
    if any(key in reason for key in THROTTLED_REASON_KEYS):
        return "throttled"
    if any(key in reason for key in BLOCKED_REASON_KEYS):
        return "blocked"
    print(f"Warning: unrecognised pending reason {reason!r}, counting it as blocked")
    return "blocked"


def get_partition_gpu_types():
    """Map every partition to the set of GPU types its nodes provide."""
    rows = parse_cmd("sinfo -a -o '%5000P|%5000G' --noheader")
    # Same shape as the gres string parsed in slurm_gpustat.parse_all_gpus.
    gres_pattern = re.compile(r"gpu:(?:([^:]*):)?(\d*)(?:\(\S*\))?")
    partition_types = defaultdict(set)
    for row in rows:
        tokens = row.split("|")
        if len(tokens) < 2:
            print(f"Warning: unexpected sinfo row {row!r}, skipping")
            continue
        # sinfo marks the default partition with a trailing '*'
        partition = tokens[0].strip().rstrip("*")
        for entry in tokens[1].strip().split(","):
            if not entry.startswith("gpu"):
                continue
            match = gres_pattern.search(entry)
            if not match:
                print(f"Warning: could not parse gres {entry!r} of partition {partition}, skipping")
                continue
            partition_types[partition].add(match.group(1) or "NONAME_GPU")
    return partition_types


def parse_queue_stats_to_table():
    """Aggregate all pending GPU jobs by the GPU type they are actually waiting for.

    A pending job may list several partitions and may not name a GPU type at all, so
    the types it can land on are derived from the partitions it requested. Jobs left
    with a single candidate type are committed demand for that type; jobs with several
    candidates are reported separately, because they will only ever consume one of
    them. Jobs are also split by their pending reason, so that the queue depth that
    actually competes for GPUs is not drowned out by jobs held back by dependencies.
    """
    partition_types = get_partition_gpu_types()
    gpu_req_pattern = re.compile(r"gres/gpu(?::([^:]+))?:(\d+)")
    rows = parse_cmd("squeue -a -t PENDING --noheader -o '%i|%P|%b|%D|%r'")

    def _new_bucket():
        return {category: {"jobs": 0, "gpus": 0} for category in QUEUE_CATEGORIES}

    committed = defaultdict(_new_bucket)  # gpu type -> category -> counts
    flexible = defaultdict(_new_bucket)  # gpu type -> category -> counts
    totals = _new_bucket()
    # {partition_group: {"jobs": N, "gpus": N, "types": {type: count}}}
    group_stats = defaultdict(lambda: {"jobs": 0, "gpus": 0, "types": defaultdict(int)})
    unknown_partitions = set()
    impossible_requests = set()

    for row in rows:
        fields = row.strip().split("|", 4)
        if len(fields) < 5:
            print(f"Warning: unexpected squeue row {row!r}, skipping")
            continue
        jobid, partitions, tres, num_nodes_str, reason = fields

        partition_list = [p.strip() for p in partitions.split(",")]
        gpu_parts = [p for p in partition_list if _is_gpu_partition(p)]
        if not gpu_parts:
            continue

        match = gpu_req_pattern.search(tres)
        if not match:
            continue

        req_type = match.group(1)
        num_nodes = int(num_nodes_str.strip()) if num_nodes_str.strip().isdigit() else 1
        num_tasks = _count_array_tasks(jobid)
        total_gpus = int(match.group(2)) * num_nodes * num_tasks

        candidates = set()
        for partition in gpu_parts:
            if partition not in partition_types:
                unknown_partitions.add(partition)
                continue
            candidates |= partition_types[partition]
        if req_type:
            narrowed = {t for t in candidates if t == req_type}
            if not narrowed:
                # The named type is offered by none of the requested partitions; keep
                # the request visible instead of silently dropping the job.
                impossible_requests.add((", ".join(sorted(gpu_parts)), req_type))
                narrowed = {req_type}
            candidates = narrowed
        if not candidates:
            print(
                f"Warning: pending job {jobid} asks for GPUs in {gpu_parts} but no GPU "
                "type could be resolved, skipping"
            )
            continue

        category = _classify_pending_reason(reason)
        totals[category]["jobs"] += num_tasks
        totals[category]["gpus"] += total_gpus

        target = committed if len(candidates) == 1 else flexible
        for gpu_type in candidates:
            target[gpu_type][category]["jobs"] += num_tasks
            target[gpu_type][category]["gpus"] += total_gpus

        group_key = ", ".join(sorted(gpu_parts))
        group_stats[group_key]["jobs"] += num_tasks
        group_stats[group_key]["gpus"] += total_gpus
        group_stats[group_key]["types"][req_type if req_type else "any"] += total_gpus

    if unknown_partitions:
        print(
            f"Warning: no GPU inventory found for partition(s) {sorted(unknown_partitions)}; "
            "pending jobs there were resolved from their other partitions only"
        )
    for group, req_type in sorted(impossible_requests):
        print(
            f"Warning: pending jobs ask for gpu type {req_type!r} in partition(s) "
            f"{group}, which advertise no such GPU"
        )

    if not group_stats:
        return "<p>No pending GPU jobs.</p>"

    pools = _gpu_pools()
    total_by_type = pools["total_by_type"]
    free_by_type = pools["free_by_type"]

    all_types = set(committed) | set(flexible) | set(total_by_type)
    type_list = sorted(all_types, key=_type_rank)
    gpu_type_list = [t for t in type_list if not _is_mig_type(t)]
    mig_type_list = [t for t in type_list if _is_mig_type(t)]

    def _pressure_cell(wanted, capacity):
        """Report queued GPUs as a multiple of the partition's whole capacity."""
        if capacity == 0:
            return '<span class="queue-muted">n/a</span>'
        if wanted == 0:
            return '<span class="queue-muted">0&times;</span>'
        ratio = wanted / capacity
        if ratio >= 1:
            css = "queue-pressure-high"
        elif ratio >= 0.25:
            css = "queue-pressure-mid"
        else:
            css = "queue-pressure-low"
        return f'<span class="{css}">{ratio:.1f}&times;</span>'

    def _render_type_rows(types_to_render):
        html = ""
        for gpu_type in types_to_render:
            own = committed[gpu_type]
            flex = flexible[gpu_type]
            capacity = sum(x["count"] for x in total_by_type.get(gpu_type, []))
            free = sum(x["count"] for x in free_by_type.get(gpu_type, []))
            waiting_jobs = own["waiting"]["jobs"]
            waiting_gpus = own["waiting"]["gpus"]
            flex_jobs = flex["waiting"]["jobs"]
            flex_gpus = flex["waiting"]["gpus"]
            throttled_gpus = own["throttled"]["gpus"]
            blocked_gpus = own["blocked"]["gpus"]

            if flex_gpus:
                flex_cell = (
                    f'<span class="queue-flex" title="These jobs also requested other '
                    f'GPU types and will consume only one of them">+{flex_jobs} jobs / '
                    f"{flex_gpus} gpus</span>"
                )
            else:
                flex_cell = '<span class="queue-muted">&ndash;</span>'

            idle_gpus = throttled_gpus + blocked_gpus
            if idle_gpus:
                idle_cell = (
                    f'<span class="queue-muted" title="{throttled_gpus} gpus held back by '
                    f"QOS/array limits, {blocked_gpus} gpus blocked by dependencies, holds "
                    f'or bad constraints">{idle_gpus} gpus</span>'
                )
            else:
                idle_cell = '<span class="queue-muted">&ndash;</span>'

            html += (
                f"<tr><td><b>{gpu_type}</b> {GMEM.get(gpu_type, '')}</td>"
                f"<td>{get_resource_bar(free, capacity, text=f'{free} / {capacity}')}</td>"
                f"<td>{waiting_jobs}</td><td><b>{waiting_gpus}</b></td>"
                f"<td>{_pressure_cell(waiting_gpus, capacity)}</td>"
                f"<td>{flex_cell}</td><td>{idle_cell}</td></tr>"
            )
        return html

    header = (
        "<tr><th>GPU Type</th><th>Free Now</th><th>Waiting Jobs</th><th>GPUs Wanted</th>"
        '<th title="Queued GPUs divided by the total number of GPUs of this type">'
        "Queue Depth</th>"
        '<th title="Jobs that could also run on another GPU type">Flexible</th>'
        '<th title="Pending jobs that are not competing for a GPU right now">'
        "Not Competing</th></tr>"
    )

    summary = (
        f'<tr><td colspan="7"><h3>Competing for GPUs now: '
        f'{totals["waiting"]["jobs"]} jobs / {totals["waiting"]["gpus"]} gpus'
        f'<span class="queue-muted"> &nbsp;|&nbsp; throttled by QOS/array limits: '
        f'{totals["throttled"]["jobs"]} jobs / {totals["throttled"]["gpus"]} gpus '
        f"&nbsp;|&nbsp; blocked by dependencies or holds: "
        f'{totals["blocked"]["jobs"]} jobs / {totals["blocked"]["gpus"]} gpus</span>'
        "</h3></td></tr>"
    )

    table_html = f"<table>{summary}{header}{_render_type_rows(gpu_type_list)}</table>"
    table_html += (
        '<p class="queue-note">A job that names several partitions is counted under '
        "every GPU type it could land on, so the <b>Flexible</b> column overlaps between "
        "rows; only the summary line above is a plain per-job total.</p>"
    )

    if mig_type_list:
        table_html += (
            '<details class="mig-details"><summary>MIG slices</summary>'
            f"<table>{header}{_render_type_rows(mig_type_list)}</table></details>"
        )

    groups = sorted(group_stats.items(), key=lambda x: x[1]["gpus"], reverse=True)
    group_rows = ""
    for group, data in groups:
        type_parts = []
        for gpu_type, count in sorted(data["types"].items(), key=lambda x: x[1], reverse=True):
            if gpu_type == "any":
                type_parts.append(f'<span class="gpu-type-any">{gpu_type}&times;{count}</span>')
            else:
                type_parts.append(f"{gpu_type}&times;{count}")
        group_rows += (
            f"<tr><td>{group}</td><td>{data['jobs']}</td><td>{data['gpus']}</td>"
            f"<td>{', '.join(type_parts)}</td></tr>"
        )
    all_jobs = sum(v["jobs"] for v in totals.values())
    all_gpus = sum(v["gpus"] for v in totals.values())
    table_html += (
        '<details class="mig-details"><summary>Raw breakdown by requested partitions'
        f" ({all_jobs} jobs, {all_gpus} gpus)</summary><table>"
        "<tr><th>Partition(s)</th><th>Pending Jobs</th><th>GPUs Needed</th>"
        "<th>Requested GPU Type</th></tr>"
        f"{group_rows}"
        f'<tr class="queue-stats-total"><td><b>Total</b></td><td><b>{all_jobs}</b></td>'
        f"<td><b>{all_gpus}</b></td><td></td></tr></table></details>"
    )

    return table_html


# ---------------------------------------------------------------- wait estimate
# A "standard job" is one GPU plus the share of the node's CPUs and memory that
# one GPU is entitled to, i.e. node_cpus / node_gpus and node_mem / node_gpus.
# Asking SLURM to schedule that job with --test-only gives the real backfill
# estimate, so no priority arithmetic has to be reimplemented here. Every field
# can be overridden from the web form; the defaults below are what the form
# starts with. The account is not one of them -- it comes from the page's
# account picker, so that every panel is talking about the same account.
STANDARD_JOB_HOURS = int(os.environ.get("RIVANNA_STANDARD_HOURS", "24"))
# Guard rails for the values arriving from the browser.
STANDARD_JOB_MAX_GPUS = 64
STANDARD_JOB_MAX_CPUS = 512
STANDARD_JOB_MAX_MEM_GB = 4096
STANDARD_JOB_MAX_HOURS = 24 * 14
ACCOUNT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


# sbatch phrasings that all mean "this account has no node it could ever land on
# in this pool" -- typically another group's reservation, or a partition we are
# not allowed into. These are a permanent property of the pool rather than a
# problem with the requested job, so they are reported once instead of per row.
NO_ACCESS_MARKERS = (
    "Requested node configuration is not available",
    "Invalid partition name",
    "User's group not permitted to use this partition",
    "Job's account not permitted to use this partition",
    "Invalid account",
    "Access/permission denied",
)
# The one no-access reason that is about the account rather than the pool: it is
# what sbatch says when the submitter is not a member of the account at all.
NOT_A_MEMBER_MARKER = "Invalid account"
# Pools already reported as inaccessible, so the log is not repeated on every refresh.
_reported_no_access = set()
# Whether an account may use a pool is account, QOS and partition configuration,
# which barely moves between refreshes, yet the panel asked slurmctld to simulate
# the same fourteen jobs on every one.  One marker above does follow live state:
# "Requested node configuration is not available" also fires while every node of
# a type is drained, so the verdict is kept briefly rather than for the life of
# the process, and a pool that comes back is picked up within the TTL.
TEST_ONLY_CACHE_SECONDS = int(os.environ.get("RIVANNA_TEST_ONLY_TTL", "60"))
_test_only_cache = {}
# Union partitions already reported as skipped, for the same reason.
_reported_unions = set()


class InvalidJobRequest(ValueError):
    """Raised when the browser sends a standard-job field we refuse to run."""


def _int_arg(args, name, default, minimum, maximum, allow_auto=False):
    """Read one integer form field, rejecting anything outside its range."""
    raw = args.get(name, "").strip()
    if not raw:
        return default
    if allow_auto and raw.lower() == "auto":
        return None
    if not raw.isdigit():
        suffix = " or 'auto'" if allow_auto else ""
        raise InvalidJobRequest(f"{name} must be a whole number{suffix}, got {raw!r}")
    value = int(raw)
    if not minimum <= value <= maximum:
        raise InvalidJobRequest(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def parse_standard_job_args(args):
    """Turn the web form's query string into validated standard-job settings."""
    account = selected_account(args)
    if not ACCOUNT_PATTERN.match(account):
        raise InvalidJobRequest(f"account {account!r} is not a valid SLURM account name")
    return {
        "gpus": _int_arg(args, "gpus", 1, 1, STANDARD_JOB_MAX_GPUS),
        # None means "derive this GPU's even share of the node" per partition.
        "cpus": _int_arg(args, "cpus", None, 1, STANDARD_JOB_MAX_CPUS, allow_auto=True),
        "mem_gb": _int_arg(args, "mem", None, 1, STANDARD_JOB_MAX_MEM_GB, allow_auto=True),
        "hours": _int_arg(args, "hours", STANDARD_JOB_HOURS, 1, STANDARD_JOB_MAX_HOURS),
        "account": account,
    }


def _slurm_time_to_minutes(text):
    """Convert a SLURM time limit ('3-00:00:00', '12:00:00', '30') to minutes."""
    text = text.strip()
    if text in ("UNLIMITED", "INFINITE", "NONE"):
        return None
    days = 0
    if "-" in text:
        day_str, text = text.split("-", 1)
        if not day_str.isdigit():
            print(f"Warning: unparsable SLURM time limit {text!r}, ignoring it")
            return None
        days = int(day_str)
    parts = text.split(":")
    if not all(p.isdigit() for p in parts) or len(parts) > 3:
        print(f"Warning: unparsable SLURM time limit {text!r}, ignoring it")
        return None
    nums = [int(p) for p in parts]
    if len(nums) == 3:
        minutes = nums[0] * 60 + nums[1] + (1 if nums[2] else 0)
    elif len(nums) == 2:
        minutes = nums[0] + (1 if nums[1] else 0)
    else:
        minutes = nums[0]
    return days * 24 * 60 + minutes


def _partition_limits():
    """Read MaxTime and MaxMemPerCPU for every partition, hidden ones included."""
    rows = parse_cmd("scontrol -a show partition -o")
    limits = {}
    for row in rows:
        name = re.search(r"PartitionName=(\S+)", row)
        if not name:
            print(f"Warning: unexpected scontrol partition row {row[:60]!r}, skipping")
            continue
        max_time = re.search(r"MaxTime=(\S+)", row)
        max_mem = re.search(r"MaxMemPerCPU=(\d+)", row)
        limits[name.group(1)] = {
            "max_minutes": _slurm_time_to_minutes(max_time.group(1)) if max_time else None,
            "max_mem_per_cpu": int(max_mem.group(1)) if max_mem else None,
        }
    return limits


def _partition_nodes():
    """Map every monitored GPU partition to the set of nodes it contains."""
    rows = parse_cmd(f"sinfo -h -p {MONITORED_PARTITIONS} -o '%P|%N'")
    nodes = defaultdict(set)
    for row in rows:
        tokens = row.split("|")
        if len(tokens) < 2:
            print(f"Warning: unexpected sinfo node row {row!r}, skipping")
            continue
        nodes[tokens[0].strip().rstrip("*")] |= set(parse_node_names(tokens[1].strip()))
    return nodes


def _standard_job_specs(job):
    """Build the standard job for every monitored (partition, GPU type) pool.

    Rows are keyed by partition rather than by GPU type because pools of the same
    type can queue very differently (``gpu-a100-40`` vs ``gpu-a100-80``). Union
    partitions such as ``gpu`` are dropped when every type they offer is already
    covered by a narrower partition, so each pool is listed once.

    ``job`` comes from :func:`parse_standard_job_args`. A ``cpus`` or ``mem_gb``
    of ``None`` means "one GPU's even share of this partition's node", which is
    why those two are resolved per partition rather than once up front.
    """
    rows = parse_cmd(f"sinfo -h -p {MONITORED_PARTITIONS} -o '%P|%c|%m|%G'")
    gres_pattern = re.compile(r"gpu:(?:([^:]*):)?(\d+)")
    # partition -> gpu type -> densest (cpus, mem, gpus) node shape offering it
    by_partition = defaultdict(dict)
    for row in rows:
        tokens = row.split("|")
        if len(tokens) < 4:
            print(f"Warning: unexpected sinfo row {row!r}, skipping")
            continue
        partition = tokens[0].strip().rstrip("*")
        # sinfo marks values that differ across the listed nodes with a trailing '+'
        cpu_str = tokens[1].strip().rstrip("+")
        mem_str = tokens[2].strip().rstrip("+")
        if not cpu_str.isdigit() or not mem_str.isdigit():
            print(f"Warning: unparsable cpu/mem {cpu_str!r}/{mem_str!r} in {partition}, skipping")
            continue
        for entry in tokens[3].strip().split(","):
            if not entry.startswith("gpu"):
                continue
            match = gres_pattern.search(entry)
            if not match:
                print(f"Warning: could not parse gres {entry!r} of partition {partition}, skipping")
                continue
            gpu_type = match.group(1) or "NONAME_GPU"
            n_gpu = int(match.group(2))
            if n_gpu == 0:
                print(f"Warning: partition {partition} reports 0 {gpu_type} gpus, skipping")
                continue
            # Keep the densest node: it defines the share one GPU is entitled to.
            if by_partition[partition].get(gpu_type, (0, 0, 0))[2] < n_gpu:
                by_partition[partition][gpu_type] = (int(cpu_str), int(mem_str), n_gpu)

    # A type is "covered" by any partition that offers fewer types than this one.
    narrower = defaultdict(set)
    for partition, types in by_partition.items():
        for gpu_type in types:
            narrower[gpu_type].add(len(types))
    unions = set()
    for partition, types in by_partition.items():
        if len(types) > 1 and all(min(narrower[t]) < len(types) for t in types):
            unions.add(partition)
    for partition in sorted(unions - _reported_unions):
        _reported_unions.add(partition)
        print(f"Note: skipping union partition {partition}, its GPU types are listed separately")

    limits = _partition_limits()
    specs = []
    for partition, types in sorted(by_partition.items()):
        if partition in unions:
            continue
        limit = limits.get(partition, {})
        for gpu_type, (cpus, mem, n_gpu) in sorted(types.items()):
            capped = []
            if job["cpus"] is None:
                job_cpus = max(cpus // n_gpu, 1) * job["gpus"]
            else:
                job_cpus = job["cpus"]
            if job["mem_gb"] is None:
                job_mem = max(mem // n_gpu, 1000) * job["gpus"]
            else:
                job_mem = job["mem_gb"] * 1000

            max_mem_per_cpu = limit.get("max_mem_per_cpu")
            if max_mem_per_cpu and job_mem > max_mem_per_cpu * job_cpus:
                job_mem = max_mem_per_cpu * job_cpus
                capped.append(f"memory capped by MaxMemPerCPU={max_mem_per_cpu}M")
            minutes = job["hours"] * 60
            if limit.get("max_minutes") and minutes > limit["max_minutes"]:
                minutes = limit["max_minutes"]
                capped.append(f"walltime capped by the partition MaxTime of {minutes} min")

            specs.append(
                {
                    "partition": partition,
                    "gpu_type": gpu_type,
                    "gpus": job["gpus"],
                    "cpus": job_cpus,
                    "mem_mb": job_mem,
                    "minutes": minutes,
                    "capped": capped,
                }
            )
    return specs


def _test_only_check(spec, account):
    """Ask SLURM whether this job shape is admissible for ``account`` in this pool.

    Only the verdict is used. The start time sbatch prints is discarded on
    purpose: slurmctld derives it (``_delayed_job_start_time``) by adding up the
    CPU-time of every pending job in the partition whose priority number is at
    least ours, *including* jobs held by a dependency, and spreading that over
    the partition. In a busy pool that adds weeks to a queue that is really a
    few days long, which is why :func:`_plan_pool` estimates the start
    instead. Returns None when the job is admissible, else a
    ``("no_access" | "error", reason)`` tuple.
    """
    # Every interpolated field is validated in parse_standard_job_args or read
    # straight out of sinfo, so none of them can carry shell syntax.
    cmd = (
        f"sbatch --test-only -A {account} -p {spec['partition']} "
        f"--gres=gpu:{spec['gpu_type']}:{spec['gpus']} -c {spec['cpus']} "
        f"--mem={spec['mem_mb']}M -t {spec['minutes']} --wrap=hostname"
    )
    key = (
        account, spec["partition"], spec["gpu_type"],
        spec["gpus"], spec["cpus"], spec["mem_mb"], spec["minutes"],
    )
    now = time.monotonic()
    cached = _test_only_cache.get(key)
    if cached is not None and now - cached[0] < TEST_ONLY_CACHE_SECONDS:
        return cached[1]

    try:
        out = check_output(cmd, shell=True, stderr=STDOUT).decode("utf-8")
    except CalledProcessError as exc:
        out = exc.output.decode("utf-8")
    if re.search(r"to start at (\S+)", out):
        verdict = None
    else:
        lines = out.strip().splitlines()
        reason = lines[-1].replace("sbatch: error: ", "") if lines else "no output"
        pool = (spec["partition"], spec["gpu_type"])
        if any(marker in reason for marker in NO_ACCESS_MARKERS):
            if pool not in _reported_no_access:
                _reported_no_access.add(pool)
                print(
                    f"Note: {account} has no schedulable node for {spec['gpu_type']} in "
                    f"{spec['partition']} ({reason}); leaving that pool out of the estimates"
                )
            verdict = ("no_access", reason)
        else:
            print(f"Warning: {spec['gpu_type']} on {spec['partition']} rejects this job shape: {reason}")
            verdict = ("error", reason)
    _test_only_cache[key] = (now, verdict)
    return verdict


# ------------------------------------------------------------ queue snapshot
# squeue --Format pads every column to exactly the width asked for (and truncates
# anything longer), so rows are sliced by position rather than split on
# whitespace: nodelists have no spaces, but reasons do.
QUEUE_FORMAT = (
    ("jobid", 32),
    ("arrayjobid", 14),
    ("arraytaskid", 28),
    ("prioritylong", 14),
    ("partition", 200),
    ("account", 32),
    ("state", 14),
    ("endtime", 22),
    ("nodelist", 400),
    ("numnodes", 10),
    ("timelimit", 14),
    ("tres-per-node", 64),
    ("tres-per-job", 48),
    ("tres-per-task", 48),
    ("tres-alloc", 300),
    ("reason", 120),
)
# Arrays with more pending tasks than this are cut down, with a warning: the
# simulation would otherwise spend its time on a queue that cannot run anyway.
ARRAY_TASK_CAP = 2000
# Resolution and reach of the queue replay, see _plan_pool.
PLAN_MINUTES = 10
PLAN_HORIZON_DAYS = 30


def _squeue_rows(extra_args):
    """Run squeue with QUEUE_FORMAT and return one dict per job."""
    fmt = ",".join(f"{name}:{width}" for name, width in QUEUE_FORMAT)
    rows = parse_cmd(f"squeue -a -h {extra_args} -O '{fmt}'")
    entries = []
    for row in rows:
        entry = {}
        pos = 0
        for name, width in QUEUE_FORMAT:
            entry[name] = row[pos : pos + width].strip()
            pos += width
        if not entry["jobid"]:
            print(f"Warning: squeue row without a job id {row[:80]!r}, skipping")
            continue
        entries.append(entry)
    return entries


def _tres_field(tres, name):
    """Value of ``name`` in a TRES string like ``cpu=8,mem=64G,gres/gpu=2``."""
    match = re.search(rf"(?:^|,){re.escape(name)}=([^,]+)", tres)
    return match.group(1) if match else None


def _mem_to_mb(text):
    """``48G`` / ``4000000M`` / ``7812.50G`` / ``2T`` to megabytes."""
    match = re.fullmatch(r"([0-9.]+)([KMGT]?)", text.strip())
    if not match:
        return None
    value = float(match.group(1))
    scale = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[match.group(2)]
    return int(value * scale)


def _gres_gpu_type(*tres_strings):
    """GPU type named in any of the tres-per-* strings, or None when untyped."""
    for text in tres_strings:
        match = re.search(r"gpu:([^:,()]+):\d+", text)
        if match:
            return match.group(1)
    return None


def _expand_array_tasks(task_spec):
    """Task count and throttle of a pending array spec such as ``53-199%12``."""
    throttle = None
    if "%" in task_spec:
        task_spec, throttle_str = task_spec.split("%", 1)
        if throttle_str.isdigit():
            throttle = int(throttle_str)
        else:
            print(f"Warning: unparsable array throttle in {task_spec!r}%{throttle_str!r}")
    count = 0
    for piece in task_spec.split(","):
        piece = piece.strip()
        step = 1
        if ":" in piece:
            piece, step_str = piece.split(":", 1)
            if step_str.isdigit() and int(step_str) > 0:
                step = int(step_str)
        if "-" in piece:
            lo, hi = piece.split("-", 1)
            if lo.isdigit() and hi.isdigit():
                count += max((int(hi) - int(lo)) // step + 1, 0)
                continue
        elif piece.isdigit():
            count += 1
            continue
        print(f"Warning: unparsable array task spec {piece!r}, counting it as one task")
        count += 1
    return count, throttle


def _sprio_by_job():
    """jobid -> {partition: total priority} for every job SLURM is ranking.

    Multi-partition jobs get one row per partition, each with the priority the
    job has *there*. Jobs held by a dependency have no priority and are absent,
    which is exactly the set of jobs not competing for a node right now.
    """
    priorities = defaultdict(dict)
    for row in parse_cmd("sprio -h -o '%i|%r|%Y'"):
        fields = row.split("|")
        if len(fields) < 3:
            print(f"Warning: unexpected sprio row {row!r}, skipping")
            continue
        jobid, partition, total = (f.strip() for f in fields[:3])
        if not total.isdigit():
            print(f"Warning: unparsable sprio priority in {row!r}, skipping")
            continue
        priorities[jobid][partition] = int(total)
    return priorities


def _partition_factors():
    """Partition priority factor of every partition, as sprio would report it.

    ``PriorityJobFactor`` is normalised against the largest one on the cluster
    and scaled by ``PriorityWeightPartition``.
    """
    config = parse_cmd("scontrol show config", split=False)
    match = re.search(r"PriorityWeightPartition\s*=\s*(\d+)", config)
    if not match:
        print("Warning: PriorityWeightPartition not in scontrol show config, treating it as 0")
        return {}
    weight = int(match.group(1))
    raw = {}
    for row in parse_cmd("scontrol -a show partition -o"):
        name = re.search(r"PartitionName=(\S+)", row)
        factor = re.search(r"PriorityJobFactor=(\d+)", row)
        if not name or not factor:
            print(f"Warning: partition row without name/factor {row[:60]!r}, skipping")
            continue
        raw[name.group(1)] = int(factor.group(1))
    biggest = max(raw.values()) if raw else 0
    if not biggest:
        return {name: 0 for name in raw}
    return {name: int(value / biggest * weight) for name, value in raw.items()}


def _node_shapes():
    """node -> {"cpus", "mem_mb"} for every node of the monitored partitions."""
    shapes = {}
    for row in parse_cmd(f"sinfo -h -N -p {MONITORED_PARTITIONS} -o '%N|%c|%m'"):
        fields = row.split("|")
        if len(fields) < 3:
            print(f"Warning: unexpected sinfo node row {row!r}, skipping")
            continue
        node, cpus, mem = (f.strip() for f in fields[:3])
        if not cpus.isdigit() or not mem.isdigit():
            print(f"Warning: unparsable cpus/mem {cpus!r}/{mem!r} for node {node}, skipping")
            continue
        shapes[node] = {"cpus": int(cpus), "mem_mb": int(mem)}
    return shapes


def _first_int(text):
    """Leading integer of ``2``, ``2-4`` or ``1+``; None when there is none."""
    match = re.match(r"\d+", text.strip())
    return int(match.group(0)) if match else None


def _running_jobs_by_node(rows, now, max_minutes):
    """node -> list of running allocations on it, from squeue rows.

    Each allocation carries what the job holds on *that* node (a multi-node job
    is split evenly), when SLURM will kill it, and how much of its limit it has
    already used so the Typical estimate can shorten it.
    """
    by_node = defaultdict(list)
    for row in rows:
        gpus = _tres_field(row["tres-alloc"], "gres/gpu")
        cpus = _tres_field(row["tres-alloc"], "cpu")
        mem = _tres_field(row["tres-alloc"], "mem")
        nodes = parse_node_names(row["nodelist"]) if row["nodelist"] else []
        if not nodes:
            print(f"Warning: running job {row['jobid']} has no node list, skipping")
            continue
        n_nodes = len(nodes)
        limit = _slurm_time_to_minutes(row["timelimit"])
        if limit is None:
            limit = max_minutes
        if row["endtime"] in ("N/A", "UNKNOWN", ""):
            print(f"Warning: running job {row['jobid']} has no end time, assuming its full limit")
            end = now + timedelta(minutes=limit)
        else:
            try:
                end = datetime.strptime(row["endtime"], "%Y-%m-%dT%H:%M:%S")
            except ValueError as err:
                print(f"Warning: unparsable end time for job {row['jobid']}: {err}, assuming its limit")
                end = now + timedelta(minutes=limit)
        mem_mb = _mem_to_mb(mem) if mem else 0
        if mem and mem_mb is None:
            print(f"Warning: unparsable memory {mem!r} for job {row['jobid']}, counting 0")
            mem_mb = 0
        alloc = {
            "id": row["jobid"],
            "gpus": int(gpus) // n_nodes if gpus and gpus.isdigit() else 0,
            "cpus": int(cpus) // n_nodes if cpus and cpus.isdigit() else 0,
            "mem_mb": mem_mb // n_nodes,
            "gpu_type": _gres_gpu_type(row["tres-per-node"], row["tres-per-job"], row["tres-per-task"]),
            "end": end,
            "limit": limit,
            "array": row["arrayjobid"] if row["arrayjobid"] not in ("", "N/A") else None,
        }
        for node in nodes:
            by_node[node].append(alloc)
    return by_node


def _competing_jobs(rows, priorities, partitions, score, gpu_type, node_gpus, max_minutes):
    """Pending jobs that outrank ``score`` in any of ``partitions`` and fit this pool.

    ``rows`` are pending squeue rows, ``priorities`` comes from :func:`_sprio_by_job`.
    Array tasks still folded into their parent record are expanded so each
    counts once. Returns the list sorted by priority, highest first.
    """
    jobs = []
    for row in rows:
        ranks = priorities.get(row["jobid"], {})
        prio = max((ranks[p] for p in partitions if p in ranks), default=None)
        if prio is None or prio < score:
            continue
        reason = row["reason"]
        category = _classify_pending_reason(reason)
        if category == "blocked":
            continue
        if category == "throttled" and "JobArrayTaskLimit" not in reason:
            # Capped by a per-user/QOS limit: it starts when its owner's other
            # jobs end, which nothing here can predict.
            continue
        wanted_type = _gres_gpu_type(row["tres-per-node"], row["tres-per-job"], row["tres-per-task"])
        if wanted_type and wanted_type != gpu_type:
            continue
        n_nodes = _first_int(row["numnodes"]) or 1
        gpus = _tres_field(row["tres-alloc"], "gres/gpu")
        cpus = _tres_field(row["tres-alloc"], "cpu")
        mem = _tres_field(row["tres-alloc"], "mem")
        gpn = -(-int(gpus) // n_nodes) if gpus and gpus.isdigit() else 0
        if gpn > node_gpus:
            # Asks for more GPUs per node than this pool has: it will land elsewhere.
            continue
        mem_mb = _mem_to_mb(mem) if mem else 0
        if mem and mem_mb is None:
            print(f"Warning: unparsable memory {mem!r} for pending job {row['jobid']}, counting 0")
            mem_mb = 0
        limit = _slurm_time_to_minutes(row["timelimit"])
        if limit is None:
            limit = max_minutes
        count, throttle = 1, None
        array = row["arrayjobid"] if row["arrayjobid"] not in ("", "N/A") else None
        if array and row["jobid"] == array and row["arraytaskid"] not in ("", "N/A"):
            count, throttle = _expand_array_tasks(row["arraytaskid"])
            if count > ARRAY_TASK_CAP:
                print(
                    f"Warning: array {array} has {count} pending tasks, only the first "
                    f"{ARRAY_TASK_CAP} are simulated"
                )
                count = ARRAY_TASK_CAP
        job = {
            "id": row["jobid"],
            "prio": prio,
            "nodes": n_nodes,
            "gpn": gpn,
            "cpn": -(-int(cpus) // n_nodes) if cpus and cpus.isdigit() else 0,
            "mpn": -(-mem_mb // n_nodes),
            "minutes": limit,
            "array": array,
            "throttle": throttle,
        }
        jobs.extend([job] * count)
    jobs.sort(key=lambda j: (-j["prio"], j["id"]))
    return jobs


def _plan_pool(nodes, running_by_node, queue, ours, now, ratio=None):
    """Plan one pool the way a backfill cycle does and return when ``ours`` starts.

    ``nodes`` maps node name to its (gpus, cpus, mem_mb) capacity, ``queue`` is
    the higher-priority pending work sorted best first, and ``ours`` is the job
    being estimated; it is planned last, as a fresh submission would be. Time is
    cut into PLAN_MINUTES buckets over PLAN_HORIZON_DAYS. Running jobs hold their
    resources until they end; then every queued job, in priority order, takes
    the earliest window in which enough nodes have its GPUs, CPUs and memory
    free for its whole time limit, and array tasks also respect their throttle.
    Lower-priority jobs fill earlier gaps only when they do not touch a window
    already claimed, which is the backfill guarantee.

    With ``ratio`` set, every job is assumed to use that fraction of its limit
    instead of all of it, which is what turns the worst case into the Typical
    column. Returns None when the job finds no window inside the horizon.
    """
    names = sorted(nodes)
    index = {name: i for i, name in enumerate(names)}
    buckets = PLAN_HORIZON_DAYS * 24 * 60 // PLAN_MINUTES
    # avail[resource, node, bucket]; resources are gpus, cpus, mem_mb.
    avail = np.zeros((3, len(names), buckets), dtype=np.int64)
    for name, cap in nodes.items():
        avail[:, index[name], :] = np.array(cap, dtype=np.int64)[:, None]
    array_use = {}  # array id -> tasks of it running in each bucket
    throttles = {job["array"]: job["throttle"] for job in queue if job["throttle"]}

    def _bucket_span(minutes):
        return max(int(-(-minutes // PLAN_MINUTES)), 1)

    for node, allocs in running_by_node.items():
        if node not in index:
            continue
        for alloc in allocs:
            remaining = (alloc["end"] - now).total_seconds() / 60
            if ratio is not None:
                elapsed = alloc["limit"] - remaining
                expected = alloc["limit"] * ratio
                # Past its usual length already: assume half of what is left.
                remaining = expected - elapsed if elapsed < expected else remaining / 2
            end = min(_bucket_span(max(remaining, 1)), buckets)
            avail[:, index[node], :end] -= np.array(
                (alloc["gpus"], alloc["cpus"], alloc["mem_mb"]), dtype=np.int64
            )[:, None]
            if alloc["array"] in throttles:
                array_use.setdefault(alloc["array"], np.zeros(buckets, dtype=np.int64))
                array_use[alloc["array"]][:end] += 1
    # A node can be oversubscribed in SLURM's own books (completing jobs); it
    # is simply full then.
    np.maximum(avail, 0, out=avail)

    for job in list(queue) + [ours]:
        minutes = job["minutes"]
        if ratio is not None and job is not ours:
            minutes = max(minutes * ratio, 1)
        span = _bucket_span(minutes)
        if span > buckets:
            if job is ours:
                return None
            continue
        ok = (
            (avail[0] >= job["gpn"]) & (avail[1] >= job["cpn"]) & (avail[2] >= job["mpn"])
        )
        if job["throttle"]:
            use = array_use.setdefault(job["array"], np.zeros(buckets, dtype=np.int64))
            ok &= (use < job["throttle"])[None, :]
        # A window fits when it holds no bad bucket: compare cumulative counts.
        bad = np.cumsum(~ok, axis=1)
        bad = np.concatenate([np.zeros((len(names), 1), dtype=bad.dtype), bad], axis=1)
        fits = (bad[:, span:] - bad[:, :-span]) == 0  # [node, start bucket]
        starts = np.nonzero(fits.sum(axis=0) >= job["nodes"])[0]
        if not len(starts):
            if job is ours:
                return None
            # Never fits inside the horizon: it cannot delay us either.
            continue
        start = int(starts[0])
        if job is ours:
            return now + timedelta(minutes=start * PLAN_MINUTES)
        candidates = np.nonzero(fits[:, start])[0]
        # Best fit: the nodes with the fewest GPUs free at that time, so that
        # whole nodes stay open for the multi-node jobs.
        chosen = sorted(candidates, key=lambda i: (avail[0, i, start], avail[1, i, start]))
        chosen = chosen[: job["nodes"]]
        avail[:, chosen, start : start + span] -= np.array(
            (job["gpn"], job["cpn"], job["mpn"]), dtype=np.int64
        )[:, None, None]
        if job["throttle"]:
            array_use[job["array"]][start : start + span] += 1
    return None


def _account_score(account, partition_factors):
    """Priority of a job submitted right now by ``account``, per partition."""
    profile, max_qos = _account_priority_profile((account,))
    if account not in profile or not max_qos:
        return None
    weights = _priority_weights()
    entry = profile[account]
    base = entry["qos_priority"] / max_qos * weights["PriorityWeightQOS"]
    base += entry["fairshare"] * weights["PriorityWeightFairShare"]
    return {p: int(base + factor) for p, factor in partition_factors.items()}


# How far back to look when measuring how much of their requested walltime jobs
# actually use, how long that measurement is reused, and the smallest sample we
# are willing to draw a conclusion from.
WALLTIME_SAMPLE_DAYS = int(os.environ.get("RIVANNA_WALLTIME_DAYS", "3"))
WALLTIME_CACHE_SECONDS = int(os.environ.get("RIVANNA_WALLTIME_TTL", "1800"))
WALLTIME_MIN_SAMPLE = 20
# Job states that mean the job is over and its Elapsed field is final.
FINISHED_STATES = (
    "COMPLETED",
    "CANCELLED",
    "FAILED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
)
# {"fetched_at": monotonic seconds, "data": {partition: {"ratio", "jobs"}}}
_walltime_cache = {"fetched_at": None, "data": {}}


def _walltime_efficiency():
    """Measure what fraction of its requested walltime a finished GPU job really uses.

    ``sbatch --test-only`` assumes every job ahead of you runs to its full time
    limit, which is why its answer is an upper bound. This ratio, measured per
    partition from the last few days of accounting data, is what turns that bound
    into a realistic figure. One sacct call covers every monitored partition, and
    the result is cached because it barely moves between page refreshes.
    """
    now = time.monotonic()
    if (
        _walltime_cache["fetched_at"] is not None
        and now - _walltime_cache["fetched_at"] < WALLTIME_CACHE_SECONDS
    ):
        return _walltime_cache["data"]

    cmd = (
        f"sacct -a -X -S now-{WALLTIME_SAMPLE_DAYS}days -r {MONITORED_PARTITIONS} "
        "-n -P -o Partition,Elapsed,Timelimit,State,AllocTRES"
    )
    try:
        rows = parse_cmd(cmd)
    except CalledProcessError as exc:
        print(f"Warning: sacct failed ({exc}); the Typical column will be blank")
        _walltime_cache.update({"fetched_at": now, "data": {}})
        return {}

    totals = defaultdict(lambda: {"elapsed": 0, "limit": 0, "jobs": 0})
    for row in rows:
        fields = row.split("|")
        if len(fields) < 5:
            print(f"Warning: unexpected sacct row {row!r}, skipping")
            continue
        partition, elapsed_str, limit_str, state, tres = fields[:5]
        if not state.startswith(FINISHED_STATES):
            continue
        if "gres/gpu" not in tres:
            continue
        elapsed = _slurm_time_to_minutes(elapsed_str)
        limit = _slurm_time_to_minutes(limit_str)
        if not limit:
            # UNLIMITED or unparsable: there is no ratio to compute against.
            continue
        if elapsed is None:
            print(f"Warning: sacct row {row!r} has no elapsed time, skipping")
            continue
        entry = totals[partition.strip()]
        entry["elapsed"] += elapsed
        entry["limit"] += limit
        entry["jobs"] += 1

    data = {}
    for partition, entry in totals.items():
        if entry["jobs"] < WALLTIME_MIN_SAMPLE or entry["limit"] == 0:
            print(
                f"Note: only {entry['jobs']} finished GPU jobs in {partition} over the last "
                f"{WALLTIME_SAMPLE_DAYS} days, too few for a Typical estimate"
            )
            continue
        # Jobs can overrun slightly into the grace period; a ratio above 1 would
        # make Typical worse than the upper bound it is scaling.
        data[partition] = {
            "ratio": min(entry["elapsed"] / entry["limit"], 1.0),
            "jobs": entry["jobs"],
        }
    _walltime_cache.update({"fetched_at": now, "data": data})
    return data


def _wait_css(seconds):
    """Colour a wait: startable now, within the working day, or worse."""
    if seconds <= 60:
        return "queue-pressure-low"
    return "queue-pressure-mid" if seconds <= 6 * 3600 else "queue-pressure-high"


def _format_delta(seconds):
    """Render a wait in the coarsest unit that still reads precisely."""
    if seconds <= 60:
        return "now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def parse_wait_estimate_to_table(job=None):
    """Estimate how long the requested job would wait in each GPU pool.

    One SLURM snapshot (running jobs, ranked pending jobs, node shapes) feeds a
    per-pool scheduler replay, see :func:`_plan_pool`. sbatch --test-only is
    still run once per pool, but only to learn whether the account may use it.
    """
    if job is None:
        job = parse_standard_job_args({})
    specs = _standard_job_specs(job)
    if not specs:
        return "<p>No GPU partitions found.</p>"

    pools = _gpu_pools()
    states = pools["states"]
    total_by_node = pools["total_by_node"]
    free_by_node = pools["free_by_node"]
    partition_nodes = _partition_nodes()
    shapes = _node_shapes()
    limits = _partition_limits()
    efficiency = _walltime_efficiency()
    factors = _partition_factors()
    scores = _account_score(job["account"], factors)
    if scores is None:
        return (
            f"<p>Could not work out the priority a job from <b>{job['account']}</b> "
            "would get, so there is nothing to estimate against.</p>"
        )
    priorities = _sprio_by_job()
    now = datetime.now()
    default_limit = max(
        (l["max_minutes"] for l in limits.values() if l["max_minutes"]), default=3 * 24 * 60
    )
    running_by_node = _running_jobs_by_node(_squeue_rows("-t RUNNING"), now, default_limit)
    pending_rows = _squeue_rows(f"-t PENDING -p {MONITORED_PARTITIONS}")

    def _count(by_node, partition, gpu_type):
        return sum(
            entry["count"]
            for node in partition_nodes.get(partition, ())
            for entry in by_node.get(node, ())
            if entry["type"] == gpu_type
        )

    rows = []
    unavailable = []
    no_access = []
    for spec in specs:
        partition, gpu_type = spec["partition"], spec["gpu_type"]
        failure = _test_only_check(spec, job["account"])
        if failure is not None:
            if failure[0] == "no_access":
                no_access.append((spec, failure[1]))
            else:
                unavailable.append((spec, failure[1]))
            continue

        # The pool: reachable nodes of this partition carrying this GPU type.
        nodes = {}
        for node in partition_nodes.get(partition, ()):
            if is_inaccessible(states.get(node, "down")):
                continue
            gpus = sum(e["count"] for e in total_by_node.get(node, ()) if e["type"] == gpu_type)
            if not gpus:
                continue
            shape = shapes.get(node)
            if shape is None:
                print(f"Warning: no cpu/mem shape for node {node}, leaving it out of {partition}")
                continue
            nodes[node] = (gpus, shape["cpus"], shape["mem_mb"])
        if not nodes:
            unavailable.append((spec, "no reachable node with this GPU type right now"))
            continue
        node_gpus = max(cap[0] for cap in nodes.values())

        # Jobs already on those nodes, whatever partition they came through.
        running = {}
        for node in nodes:
            running[node] = [
                a
                for a in running_by_node.get(node, ())
                if a["gpu_type"] in (None, gpu_type)
            ]

        # Everyone ranked above a fresh job of ours in any partition sharing
        # these nodes: those are the jobs SLURM will place before it.
        sharing = [p for p, members in partition_nodes.items() if members & nodes.keys()]
        score = scores.get(partition)
        if score is None:
            print(f"Warning: no partition factor for {partition}, ranking against 0")
            score = 0
        queue = _competing_jobs(
            pending_rows, priorities, sharing, score, gpu_type, node_gpus,
            limits.get(partition, {}).get("max_minutes") or default_limit,
        )
        our_nodes = -(-spec["gpus"] // node_gpus)
        ours = {
            "id": "ours",
            "prio": score,
            "nodes": our_nodes,
            "gpn": -(-spec["gpus"] // our_nodes),
            "cpn": -(-spec["cpus"] // our_nodes),
            "mpn": -(-spec["mem_mb"] // our_nodes),
            "minutes": spec["minutes"],
            "array": None,
            "throttle": None,
        }
        worst = _plan_pool(nodes, running, queue, ours, now)
        usage = efficiency.get(partition)
        typical = _plan_pool(nodes, running, queue, ours, now, usage["ratio"]) if usage else None
        rows.append(
            {
                "spec": spec,
                "capacity": _count(total_by_node, partition, gpu_type),
                "free": _count(free_by_node, partition, gpu_type),
                "ahead_jobs": len(queue),
                "ahead_gpus": sum(j["gpn"] * j["nodes"] for j in queue),
                "score": score,
                "worst": worst,
                "typical": typical,
                "usage": usage,
            }
        )

    far = datetime.max
    rows.sort(key=lambda r: (r["typical"] or r["worst"] or far, r["worst"] or far))

    def _wait_cell(start, title):
        if start is None:
            return (
                f'<span class="queue-pressure-high" title="Not within {PLAN_HORIZON_DAYS} '
                f'days in the replay. {title}">&gt;{PLAN_HORIZON_DAYS}d</span>'
            )
        seconds = (start - now).total_seconds()
        return f'<span class="{_wait_css(seconds)}" title="{title}">{_format_delta(seconds)}</span>'

    def _render_rows(entries):
        html = ""
        for row in entries:
            spec = row["spec"]
            gpu_type, partition = spec["gpu_type"], spec["partition"]
            hours, minutes = divmod(spec["minutes"], 60)
            walltime = f"{hours}h" if not minutes else f"{hours}h{minutes}m"
            shape = (
                f"{spec['gpus']} gpu / {spec['cpus']} cpu / "
                f"{spec['mem_mb'] // 1000}G / {walltime}"
            )
            if spec["capped"]:
                shape += (
                    f'<span class="queue-flex" title="{"; ".join(spec["capped"])}">&nbsp;*</span>'
                )
            free_text = f"{row['free']} / {row['capacity']}"
            worst_cell = _wait_cell(
                row["worst"],
                "Every job ahead of you and every running job uses its full time limit",
            )
            usage = row["usage"]
            if usage:
                typical_cell = _wait_cell(
                    row["typical"],
                    f"Every job assumed to use {usage['ratio']:.0%} of its limit, what "
                    f"{usage['jobs']} finished jobs in {partition} averaged over the last "
                    f"{WALLTIME_SAMPLE_DAYS} days",
                )
                start = row["typical"]
            else:
                typical_cell = (
                    '<span class="queue-muted" title="Not enough finished GPU jobs in '
                    f'{partition} over the last {WALLTIME_SAMPLE_DAYS} days to measure '
                    'this">&ndash;</span>'
                )
                start = row["worst"]
            start_cell = start.strftime("%m-%d %H:%M") if start else '<span class="queue-muted">&ndash;</span>'
            html += (
                f"<tr><td><b>{gpu_type}</b> {GMEM.get(gpu_type, '')}</td>"
                f"<td>{partition}</td>"
                f"<td class='queue-muted'>{shape}</td>"
                f"<td>{get_resource_bar(row['free'], row['capacity'], text=free_text)}</td>"
                f'<td title="Ranked pending jobs whose priority beats {row["score"]:,}, the '
                f'score a fresh job of yours gets in {partition}">'
                f"{row['ahead_jobs']} jobs / <b>{row['ahead_gpus']}</b> gpus</td>"
                f"<td>{worst_cell}</td>"
                f"<td><b>{typical_cell}</b></td>"
                f"<td>{start_cell}</td></tr>"
            )
        return html

    header = (
        "<tr><th>GPU Type</th><th>Partition</th>"
        "<th title=\"One GPU plus that GPU's even share of the node's cpus and memory\">"
        "Standard Job</th><th>Free Now</th>"
        '<th title="Pending jobs SLURM ranks above a fresh job of yours in this pool, '
        'and the GPUs they want. Jobs held by a dependency or a user limit are not '
        'counted">Ahead of You</th>'
        '<th title="Replay of the queue with every job running to its full time limit">'
        "Worst Case</th>"
        '<th title="Same replay with every job cut to the fraction of its limit jobs in '
        'this partition actually use">Typical</th>'
        '<th title="Start time behind the Typical column (Worst Case where Typical is '
        'unavailable)">Start</th></tr>'
    )

    mig_rows = [r for r in rows if _is_mig_type(r["spec"]["gpu_type"])]
    gpu_rows = [r for r in rows if not _is_mig_type(r["spec"]["gpu_type"])]

    table_html = (
        f"<table>{header}{_render_rows(gpu_rows)}</table>"
        '<p class="queue-note">Each row replays the scheduler for one pool: the jobs '
        f"<code>sprio</code> ranks above a fresh submission from <b>{job['account']}</b> "
        "are placed first, in priority order with backfill, as the running jobs free "
        "their GPUs, and the wait is when your job lands. Jobs held by a dependency, an "
        "invalid account or a per-user limit are left out, which is what "
        "<code>sbatch --test-only</code> gets wrong &mdash; it charges you for all of "
        "them and reports weeks in pools that clear in days. <b>Worst Case</b> lets every "
        "job run to its time limit, the promise SLURM makes. <b>Typical</b> cuts every job "
        "to the fraction of its limit that jobs in that partition actually used over the "
        f"last {WALLTIME_SAMPLE_DAYS} days (hover for the ratio and sample), which on this "
        "cluster is the number to plan around. Neither can know about jobs submitted after "
        "this page loaded, nor about per-user QOS caps. A <b>*</b> next to a job shape "
        "means the partition's own limits cut the request down; hover it for the "
        "reason.</p>"
    )
    if mig_rows:
        table_html += (
            '<details class="mig-details"><summary>MIG slices</summary>'
            f"<table>{header}{_render_rows(mig_rows)}</table></details>"
        )
    if unavailable:
        # These failed on the job shape the form asked for, so the reason is
        # actionable: shrink the request, or accept that the pool cannot host it.
        rows_html = "".join(
            f"<tr><td><b>{spec['gpu_type']}</b></td><td>{spec['partition']}</td>"
            f"<td class='queue-muted'>{error}</td></tr>"
            for spec, error in unavailable
        )
        table_html += (
            '<details class="mig-details"><summary>Pools this job shape does not fit '
            f"({len(unavailable)})</summary><table>"
            "<tr><th>GPU Type</th><th>Partition</th><th>Reason</th></tr>"
            f"{rows_html}</table></details>"
        )
    if no_access:
        # "Invalid account" is a statement about the account, not the pool, and
        # it makes every other pool's verdict beside the point; saying "another
        # group's reservation" there would send the reader looking in the wrong
        # place. It does not come back for every pool, because SLURM rejects
        # some on node configuration before it ever looks at the account.
        not_member = [reason for _, reason in no_access if NOT_A_MEMBER_MARKER in reason]
        if not_member:
            table_html += (
                f'<p class="queue-note">SLURM will not schedule anything under '
                f"<b>{job['account']}</b> for you: <code>{not_member[0]}</code>. "
                "You are most likely not a member of that account yet, so what is "
                "above is the queue as it stands, not a queue you can join.</p>"
            )
        else:
            pools_text = ", ".join(
                f"{spec['gpu_type']}/{spec['partition']}"
                for spec, _ in sorted(no_access, key=lambda p: (p[0]["partition"], p[0]["gpu_type"]))
            )
            table_html += (
                f'<p class="queue-note">Left out: {pools_text} &mdash; no node in those pools is '
                f"schedulable by <b>{job['account']}</b> (another group's reservation or a "
                "partition we are not in), so there is nothing to estimate.</p>"
            )
    return table_html


# ------------------------------------------------------------------- priority
# Where a freshly submitted job from the picked account would land in the queue
# it actually competes in. The comparison has to be made *inside* a partition:
# the partition priority factor is added to every job in that partition alike,
# so ranking against the whole cluster makes a pool's factor look like our own
# advantage or handicap when it cancels out.


def _priority_weights():
    """The multifactor weights SLURM is configured with."""
    config = parse_cmd("scontrol show config", split=False)
    weights = {}
    for name in ("PriorityWeightQOS", "PriorityWeightFairShare", "PriorityWeightAge"):
        match = re.search(rf"{name}\s*=\s*(\d+)", config)
        if not match:
            print(f"Warning: {name} not found in scontrol show config, treating it as 0")
            weights[name] = 0
            continue
        weights[name] = int(match.group(1))
    return weights


def _qos_priorities():
    """QOS name -> configured priority."""
    priorities = {}
    for row in parse_cmd("sacctmgr -n -P show qos format=Name,Priority"):
        fields = row.split("|")
        if len(fields) < 2 or not fields[1].strip().isdigit():
            print(f"Warning: unexpected sacctmgr qos row {row!r}, skipping")
            continue
        priorities[fields[0].strip()] = int(fields[1])
    return priorities


def _account_priority_profile(accounts=LAB_ACCOUNTS):
    """Fairshare factor and highest-priority QOS available to each of ``accounts``."""
    account_list = ",".join(accounts)
    shares = defaultdict(list)
    for row in parse_cmd(f"sshare -a -n -P -A {account_list} -o Account,User,FairShare"):
        fields = row.split("|")
        if len(fields) < 3:
            print(f"Warning: unexpected sshare row {row!r}, skipping")
            continue
        account, user, share = (f.strip() for f in fields[:3])
        # The account-level row carries no FairShare of its own; only users do.
        if not user or not share:
            continue
        try:
            shares[account].append(float(share))
        except ValueError:
            print(f"Warning: unparsable FairShare {share!r} for {account}/{user}, skipping")

    qos_priorities = _qos_priorities()
    account_qos = {}
    for row in parse_cmd(f"sacctmgr -n -P show assoc account={account_list} format=Account,QOS"):
        fields = row.split("|")
        if len(fields) < 2:
            print(f"Warning: unexpected sacctmgr assoc row {row!r}, skipping")
            continue
        account = fields[0].strip()
        for name in (q.strip() for q in fields[1].split(",") if q.strip()):
            if name not in qos_priorities:
                print(f"Warning: QOS {name!r} of {account} has no configured priority, skipping")
                continue
            # A job may pick any QOS it is entitled to, so the best one is the
            # one worth reporting.
            best = account_qos.get(account)
            if best is None or qos_priorities[name] > qos_priorities[best]:
                account_qos[account] = name

    max_qos = max(qos_priorities.values()) if qos_priorities else 0
    profile = {}
    for account in accounts:
        if account not in shares:
            print(f"Warning: no user of {account} has a FairShare value, skipping the account")
            continue
        if account not in account_qos:
            print(f"Warning: no usable QOS found for {account}, skipping the account")
            continue
        values = sorted(shares[account])
        profile[account] = {
            "fairshare": values[len(values) // 2],
            "fairshare_range": (values[0], values[-1]),
            "users": len(values),
            "qos": account_qos[account],
            "qos_priority": qos_priorities[account_qos[account]],
        }
    return profile, max_qos


def _partition_priority_data():
    """Per partition: the priorities of the jobs competing there, and its factor.

    Jobs blocked by a dependency get no priority from SLURM and so never appear
    here, which is what we want: the ranking is against the jobs actually racing
    us for a node right now.
    """
    rows = parse_cmd("sprio -h -o '%r|%Y|%P'")
    totals = defaultdict(list)
    factors = {}
    for row in rows:
        fields = row.split("|")
        if len(fields) < 3:
            print(f"Warning: unexpected sprio row {row!r}, skipping")
            continue
        partition, total, factor = (f.strip() for f in fields[:3])
        if not total.isdigit() or not factor.isdigit():
            print(f"Warning: unparsable sprio numbers in {row!r}, skipping")
            continue
        totals[partition].append(int(total))
        if partition in factors and factors[partition] != int(factor):
            print(
                f"Warning: partition {partition} reports two priority factors "
                f"({factors[partition]} and {factor}); keeping the first"
            )
        else:
            factors[partition] = int(factor)
    for partition in totals:
        totals[partition].sort()
    return totals, factors


def parse_priority_to_table(account=None):
    """Rank a freshly submitted job from ``account`` inside every GPU pool."""
    account = account or LAB_ACCOUNTS[0]
    profile, max_qos = _account_priority_profile((account,))
    if account not in profile or not max_qos:
        return f"<p>Could not read the priority settings of <b>{account}</b> from SLURM.</p>"
    weights = _priority_weights()
    totals, factors = _partition_priority_data()

    accounts = [account]
    partitions = [p for p in _partition_nodes() if p in totals]
    partitions.sort(key=lambda p: factors.get(p, 0), reverse=True)
    if not partitions:
        return "<p>No GPU partition currently has jobs competing for priority.</p>"

    def _score(account, partition):
        entry = profile[account]
        qos_part = entry["qos_priority"] / max_qos * weights["PriorityWeightQOS"]
        fair_part = entry["fairshare"] * weights["PriorityWeightFairShare"]
        # A job submitted right now has no age factor yet.
        return int(qos_part + fair_part + factors.get(partition, 0))

    header = (
        '<tr><th>Partition</th><th title="Pending jobs SLURM is currently ranking in '
        'this partition; jobs blocked by a dependency get no priority and are not '
        'counted">Competing</th>'
        + "".join(
            f'<th>{a}<br><span class="queue-muted">{profile[a]["qos"]} '
            f'/ fs {profile[a]["fairshare"]:.3f}</span></th>'
            for a in accounts
        )
        + "</tr>"
    )

    body = ""
    for partition in partitions:
        values = totals[partition]
        cells = ""
        for account in accounts:
            score = _score(account, partition)
            rank = bisect.bisect_left(values, score) / len(values) * 100
            if rank >= 75:
                css = "queue-pressure-low"
            elif rank >= 40:
                css = "queue-pressure-mid"
            else:
                css = "queue-pressure-high"
            cells += (
                f'<td><span class="{css}" title="priority {score:,} vs a median of '
                f'{values[len(values) // 2]:,} in {partition}">{rank:.0f}%</span></td>'
            )
        body += (
            f"<tr><td><b>{partition}</b> "
            f'<span class="queue-muted">(factor {factors.get(partition, 0):,})</span></td>'
            f"<td>{len(values)}</td>{cells}</tr>"
        )

    note = (
        '<p class="queue-note">Percentile of a job submitted <b>right now</b> among the '
        "jobs already queued <b>in that partition</b> &mdash; higher is better. Ranking "
        "against the whole cluster would be misleading, because the partition factor "
        "shown next to each pool is added to every job there alike and cancels out. "
        "Priority = QOS + fairshare + partition factor; age is worth at most "
        f"{weights['PriorityWeightAge']:,} even after weeks, so a new job is not "
        "meaningfully behind an old one. Fairshare is the median across each account's "
        "users and moves as the lab uses the cluster.</p>"
    )
    return f"<table>{header}{body}</table>{note}"

def parse_queue_to_table(account=None):
    """Pending jobs of ``account``, laid out like squeue's own columns.

    START_TIME is SLURM's own estimate (what ``squeue --start`` reports), filled
    in by the backfill scheduler; it reads N/A for jobs it has not planned, such
    as held ones. State and elapsed time are left out because every row here is
    PENDING at 0:00. Both timestamps drop the year, which only widened the table.

    The account comes from the page's picker and is one of LAB_ACCOUNTS, so it
    carries no shell syntax.
    """
    account = account or LAB_ACCOUNTS[0]
    rows = parse_cmd(f"squeue -a -h -t PENDING -A {account} -o '%i|%P|%u|%l|%D|%V|%S|%R'")
    if not rows:
        return f"no pending job in {account}"

    def _short(stamp):
        # 2026-09-17T16:15:17 -> 09-17 16:15; N/A and the like pass through.
        match = re.fullmatch(r"\d{4}-(\d\d-\d\d)T(\d\d:\d\d):\d\d", stamp)
        return f"{match.group(1)} {match.group(2)}" if match else stamp

    def _limit(text):
        # 3-00:00:00 -> 3d, 1-12:00:00 -> 1d12h, 12:00:00 -> 12h. Rounded up to
        # the hour, so a limit is never shown shorter than it really is.
        minutes = _slurm_time_to_minutes(text)
        if minutes is None:
            return text
        days, hours = divmod(-(-minutes // 60), 24)
        if not days:
            return f"{hours}h"
        return f"{days}d{hours}h" if hours else f"{days}d"

    line = "{:>10} {:>14} {:>8} {:>6} {:>5} {:>11} {:>11}  {}"
    out = [line.format("JOBID", "PARTITION", "USER", "LIMIT", "NODES",
                       "SUBMIT_TIME", "START_TIME", "NODELIST(REASON)")]
    for row in rows:
        fields = row.split("|", 7)
        if len(fields) < 8:
            print(f"Warning: unexpected squeue row {row!r} for {account}, skipping")
            continue
        jobid, partition, user, limit, nodes, submit, start, reason = fields
        # A job queued on several partitions lists all of them; cut it to the
        # column like squeue itself would rather than stretching every row.
        out.append(line.format(jobid, partition[:14], user, _limit(limit), nodes,
                               _short(submit), _short(start), reason))
    return "\n".join(out)


def parse_disk_io():
    """Measure disk reading speed, parse the output to a html table.

    Pre-requisite: create a byte file by running
    `dd if=/dev/zero of=/your/path/test.img bs=512MB count=1 oflag=dsync`."""

    # return '<p>Under maintenance. </p>'

    try:
        beegfs_ultra_read = check_output(
            "dd if=/scratch/shared/beegfs/shared-datasets/test/test.img of=/dev/null bs=512MB count=1 oflag=dsync",
            stderr=STDOUT,
            shell=True,
        ).decode("utf-8")
        beegfs_ultra_read = beegfs_ultra_read.split("\n")[-2].split(",")[-1].strip()
    except:
        beegfs_ultra_read = "N/A"

    try:
        beegfs_fast_read = check_output(
            "dd if=/scratch/shared/beegfs/htd/DATA/tmp/test.img of=/dev/null bs=512MB count=1 oflag=dsync",
            stderr=STDOUT,
            shell=True,
        ).decode("utf-8")
        beegfs_fast_read = beegfs_fast_read.split("\n")[-2].split(",")[-1].strip()
    except:
        beegfs_fast_read = "N/A"

    try:
        beegfs_normal_read = check_output(
            "dd if=/scratch/shared/beegfs/htd/tmp/test.img of=/dev/null bs=512MB count=1 oflag=dsync",
            stderr=STDOUT,
            shell=True,
        ).decode("utf-8")
        beegfs_normal_read = beegfs_normal_read.split("\n")[-2].split(",")[-1].strip()
    except:
        beegfs_normal_read = "N/A"

    try:
        work_normal_read = check_output(
            "dd if=/work/htd/Desktop_tmp/tmp/test.img of=/dev/null bs=512MB count=1 oflag=dsync",
            stderr=STDOUT,
            shell=True,
        ).decode("utf-8")
        work_normal_read = work_normal_read.split("\n")[-2].split(",")[-1].strip()
    except:
        work_normal_read = "N/A"

    summary = (
        "<tr> <td><b>{}</b></td> <td><b>{}</b></td> <td><b>{}</b></td> </tr>".format(
            "Disk", "Type", "Read Speed"
        )
    )
    summary += "<tr> <td>{}</td> <td>{}</td> <td>{}</td> </tr>".format(
        "/beegfs/shared-datasets <i>[ultra-fast-layer]</i>",
        "NVMe flash",
        beegfs_ultra_read,
    )
    summary += "<tr> <td>{}</td> <td>{}</td> <td>{}</td> </tr>".format(
        "/beegfs <i>[fast-layer]</i>", "SSD flash", beegfs_fast_read
    )
    summary += "<tr> <td>{}</td> <td>{}</td> <td>{}</td> </tr>".format(
        "/beegfs <i>[normal-layer]</i>", "HDD", beegfs_normal_read
    )
    summary += "<tr> <td>{}</td> <td>{}</td> <td>{}</td></tr>".format(
        "/work", "HDD", work_normal_read
    )
    table_html = f"<table>{summary}</table>"

    return table_html


def parse_disk_quota():
    """Run 'hdquota' command and parse the output to an HTML table."""
    try:
        output = check_output("hdquota", shell=True).decode("utf-8")
        lines = output.split('\n')
        
        # Create HTML table with headers
        headers = ["Storage Type", "Location", "Size", "Used", "Avail", "Use%"]
        table_html = "<table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
        
        # Add data rows
        for line in lines[2:]:  # Skip header and separator lines
            if not line.strip():
                continue
            
            # Split the line into parts, handling spaces correctly
            parts = line.split()
            
            # Skip lines that don't have enough parts
            if len(parts) < 9:  # Changed from 10 to 9 to handle single-word storage types
                continue
                
            try:
                # Handle storage type (could be one or two words)
                if parts[1] in ["Directory", "Project", "Standard"]:
                    storage_type = f"{parts[0]} {parts[1]}"
                    location = parts[2]
                    size = f"{parts[3]} {parts[4]}"
                    used = f"{parts[5]} {parts[6]}"
                    avail = f"{parts[7]} {parts[8]}"
                    use_percent = parts[9]
                else:
                    storage_type = parts[0]
                    location = parts[1]
                    size = f"{parts[2]} {parts[3]}"
                    used = f"{parts[4]} {parts[5]}"
                    avail = f"{parts[6]} {parts[7]}"
                    use_percent = parts[8]
                
                table_html += "<tr>"
                table_html += f"<td>{storage_type}</td>"
                table_html += f"<td>{location}</td>"
                table_html += f"<td>{size}</td>"
                table_html += f"<td>{used}</td>"
                table_html += f"<td>{avail}</td>"
                table_html += f"<td>{use_percent}</td>"
                table_html += "</tr>"
            except IndexError:
                continue
        
        table_html += "</table>"
        return table_html
    except Exception as e:
        error_msg = f"Error getting disk quota information: {str(e)}"
        return f"<p>{error_msg}</p>"


def parse_allocations_to_table(account=None):
    """Run 'allocations -a <account>' and parse the output to an HTML table.

    The command only prints the balance table to members of the account; for
    everyone else it lists the members and nothing else, which is why an empty
    table is reported as "not a member" rather than "no allocation".
    """
    account = account or LAB_ACCOUNTS[0]
    cmd = f"allocations -a {account}"
    output = parse_cmd(cmd)
    if not output:
        return f"<p>No allocation information found for {account}.</p>"

    # For a member the output is the balance table followed by the member list;
    # for everyone else only the member list is printed.  Both tables are laid
    # out the same way, so find the balance table by its own header rather than
    # by counting separators, or the member rows get read as balances.
    header_idx = next(
        (i for i, line in enumerate(output) if "Allocated" in line and "Remaining" in line),
        None,
    )
    if header_idx is None:
        return (
            f'<p class="queue-note">No allocation balance is visible for '
            f"<b>{account}</b>. The <code>allocations</code> command only shows it "
            "to members of the account.</p>"
        )

    allocation_lines = []
    for line in output[header_idx + 1 :]:
        if "------" in line:
            continue
        # The member list that follows starts with its own header.
        if "CommonName" in line or line.startswith("PI:"):
            break
        allocation_lines.append(line.rstrip("\n"))

    if not allocation_lines:
        return f"<p>No allocation data found for {account}.</p>"

    columns = [
        "Description",
        "StartTime",
        "EndTime",
        "Allocated",
        "Remaining",
        "PercentUsed",
        "Active",
    ]

    # Create HTML table
    table_html = "<table><tr>"
    for col in columns:
        table_html += f"<th>{col}</th>"
    table_html += "</tr>"

    # Parse each data row by splitting on whitespace. StartTime is a date and a
    # time, so a complete row has eight fields for the seven columns.
    for row in allocation_lines:
        parts = row.split()
        if len(parts) < 8:
            print(f"Warning: incomplete allocation row {row!r} for {account}, skipping")
            continue

        table_html += "<tr>"
        # Description
        table_html += f"<td>{parts[0]}</td>"
        # StartTime (date and time)
        table_html += f"<td>{parts[1]} {parts[2]}</td>"
        # EndTime
        table_html += f"<td>{parts[3]}</td>"
        # Allocated
        table_html += f"<td>{parts[4]}</td>"
        # Remaining
        table_html += f"<td>{parts[5]}</td>"
        # PercentUsed
        table_html += f"<td>{parts[6]}</td>"
        # Active
        table_html += f"<td>{parts[7]}</td>"
        table_html += "</tr>"

    table_html += "</table>"

    return table_html


# ---------------------------------------------------------------------------
# "My running jobs": live CPU, memory and GPU readings for the jobs belonging
# to whichever account started this app.  Only its own jobs can be read, since
# both sstat and a --jobid step are refused for someone else's allocation.
# ---------------------------------------------------------------------------

MY_JOBS_FORMAT = (
    ("jobid", 16),
    ("name", 48),
    ("partition", 32),
    ("timeused", 16),
    ("timelimit", 16),
    ("tres-alloc", 300),
)
# srun resolves the command against the submitting shell's PATH, which does not
# reach the compute node's own binaries, so both of these are absolute paths.
# The probe script sits next to this file, on the shared storage the compute
# nodes mount, so the step can read it wherever it lands.
NODE_PYTHON = "/usr/bin/python3"
GPU_PROBE_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gpu_probe.py"
)
NVIDIA_SMI = "/usr/bin/nvidia-smi"
NVIDIA_SMI_FIELDS = (
    "index,name,utilization.gpu,memory.used,memory.total,"
    "power.draw,power.limit,temperature.gpu"
)
# Each probe is a job step launched on a node that is already busy, and the
# login node forks them under strict overcommit, so the fan-out stays modest.
# Most of a probe is srun's step-launch latency rather than work, so eight of
# them in flight roughly halves the panel against four; twelve buys nothing and
# starts to jitter.  A fork that does fail is caught per job and falls back.
GPU_PROBE_WORKERS = 8
# How many seconds of the driver's buffered samples each average covers.  Passed
# to gpu_probe.py, so this is the only place it is chosen.
GPU_SAMPLE_WINDOW = 5.0
GPU_PROBE_TIMEOUT = 15
# jobid -> (cpu seconds, wall clock of that reading), so a refresh can report the
# CPU load since the previous one rather than the average over the whole run.
_CPU_SAMPLES = {}
# Closer together than this, two readings say nothing useful: jobacct_gather
# only samples the counters every 30s by default.
CPU_SAMPLE_MIN_GAP = 20


def _app_owner():
    """Login name of the account running this app, whose jobs the panel shows."""
    return pwd.getpwuid(os.getuid()).pw_name


def _slurm_time_to_seconds(text):
    """'1-02:03:04', '02:03:04', '03:04.512' or '61' to seconds, None if unparsable."""
    text = text.strip()
    if not text or text in ("UNLIMITED", "INFINITE", "NONE", "N/A"):
        return None
    days = 0
    if "-" in text:
        day_str, text = text.split("-", 1)
        if not day_str.isdigit():
            print(f"Warning: unparsable SLURM duration {text!r}, ignoring it")
            return None
        days = int(day_str)
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        print(f"Warning: unparsable SLURM duration {text!r}, ignoring it")
        return None
    if len(nums) == 3:
        seconds = nums[0] * 3600 + nums[1] * 60 + nums[2]
    elif len(nums) == 2:
        seconds = nums[0] * 60 + nums[1]
    elif len(nums) == 1:
        seconds = nums[0]
    else:
        print(f"Warning: unparsable SLURM duration {text!r}, ignoring it")
        return None
    return days * 86400 + seconds


def _alloc_gpu_count(tres):
    """GPUs in a TRES string, counting the typed form (``gres/gpu:a40=1``) too."""
    total = _tres_field(tres, "gres/gpu")
    if total is not None and total.isdigit():
        return int(total)
    typed = re.findall(r"(?:^|,)gres/gpu:[^=,]+=(\d+)", tres)
    return sum(int(n) for n in typed)


def _my_running_jobs():
    """One dict per running job of this account, in squeue's own order."""
    owner = _app_owner()
    fmt = ",".join(f"{name}:{width}" for name, width in MY_JOBS_FORMAT)
    rows = parse_cmd(f"squeue -a -h -u {owner} -t RUNNING -O '{fmt}'")
    jobs = []
    for row in rows:
        entry = {}
        pos = 0
        for name, width in MY_JOBS_FORMAT:
            entry[name] = row[pos : pos + width].strip()
            pos += width
        if not entry["jobid"]:
            print(f"Warning: squeue row without a job id {row[:80]!r}, skipping")
            continue
        jobs.append(entry)
    return jobs


def _sstat_usage(jobids):
    """Live counters from sstat for each job id, folded across its steps.

    sstat prints one line per step, so CPU time and resident memory are added up
    across the steps that run at once, while the GPU counters are taken at their
    highest: only the step holding the GPU reports anything for it, and the
    ``.batch`` shell that wraps it reports zeros.  A job with no GPU carries no
    ``gres/`` fields at all, and one whose steps have not registered yet is
    simply missing from the result.
    """
    usage = {}
    if not jobids:
        return usage
    rows = parse_cmd(
        f"sstat -a -n -P -j {','.join(jobids)} "
        "--format=JobID,TRESUsageInAve,TRESUsageInMax"
    )
    for row in rows:
        fields = row.split("|")
        if len(fields) < 3:
            print(f"Warning: unexpected sstat row {row[:80]!r}, skipping")
            continue
        step, ave, peak = fields[0], fields[1], fields[2]
        jobid = step.split(".", 1)[0]
        entry = usage.setdefault(
            jobid,
            {"cpu_seconds": 0.0, "mem_mb": 0, "peak_mem_mb": 0,
             "gpu_util": None, "gpu_mem_mb": None},
        )

        cpu = _slurm_time_to_seconds(_tres_field(ave, "cpu") or "")
        if cpu is not None:
            entry["cpu_seconds"] += cpu
        for key, tres in (("mem_mb", ave), ("peak_mem_mb", peak)):
            mem = _tres_field(tres, "mem")
            if mem is None:
                continue
            mem_mb = _mem_to_mb(mem)
            if mem_mb is None:
                print(f"Warning: unparsable memory {mem!r} for step {step}, ignoring it")
                continue
            entry[key] += mem_mb

        util = _tres_field(ave, "gres/gpuutil")
        if util is not None:
            if util.isdigit():
                entry["gpu_util"] = max(entry["gpu_util"] or 0, int(util))
            else:
                print(f"Warning: unparsable gres/gpuutil {util!r} for step {step}, ignoring it")
        gpu_mem = _tres_field(ave, "gres/gpumem")
        if gpu_mem is not None:
            gpu_mem_mb = _mem_to_mb(gpu_mem)
            if gpu_mem_mb is None:
                print(f"Warning: unparsable gres/gpumem {gpu_mem!r} for step {step}, ignoring it")
            else:
                entry["gpu_mem_mb"] = max(entry["gpu_mem_mb"] or 0, gpu_mem_mb)
    return usage


def _smi_number(text, jobid, field):
    """A number out of nvidia-smi, or None for the ``[N/A]`` it prints instead.

    Not every board reports every field -- a MIG slice has no power reading of
    its own, for one -- and nvidia-smi marks those with a bracketed word rather
    than leaving the column out, so those are expected and stay quiet.
    """
    try:
        return float(text)
    except ValueError:
        if not text.startswith("["):
            print(f"Warning: unreadable nvidia-smi {field}={text!r} for job {jobid}")
        return None


def _run_in_job(jobid, argv):
    """Run one short command inside a running job of ours, as a job step.

    The step runs alongside the job's own work (``--overlap``) and asks for no
    memory of its own, so it starts at once instead of queueing behind a job
    that already holds every byte it was given.  It does keep the job's GRES,
    which is what limits the reading to the GPUs this job was handed rather
    than every GPU on the node.  ``--immediate`` turns a step that cannot start
    into a quick failure instead of a refresh that hangs.

    This is deliberately not routed through parse_cmd: its retries would launch
    three steps for a job that has just ended, and it has no timeout.

    Returns the decoded stdout, or None after warning about why it could not.
    """
    cmd = [
        "srun", "-Q", "--immediate=5", f"--jobid={jobid}", "--overlap",
        "-n1", "-c1", "--mem-per-cpu=0",
    ] + list(argv)
    what = os.path.basename(argv[0])
    try:
        completed = run(cmd, stdout=PIPE, stderr=PIPE, timeout=GPU_PROBE_TIMEOUT)
    except TimeoutExpired:
        print(f"Warning: {what} probe of job {jobid} timed out after {GPU_PROBE_TIMEOUT}s")
        return None
    except OSError as exc:
        print(f"Warning: could not start the {what} probe of job {jobid}: {exc}")
        return None

    reason = " ".join(completed.stderr.decode("utf-8", "replace").split())
    if completed.returncode != 0:
        print(
            f"Warning: {what} probe of job {jobid} failed "
            f"(exit {completed.returncode}: {reason or 'no stderr'})"
        )
        return None
    if reason:
        # The probe warns about a reading it could not take but still prints the
        # rest, so this is worth logging without throwing the row away.
        print(f"Warning: {what} probe of job {jobid} reported: {reason}")
    return completed.stdout.decode("utf-8", "replace")


def _probe_gpu_nvml(jobid):
    """Read the job's GPUs through NVML, averages included; None if it failed.

    gpu_probe.py pulls utilisation and power out of the ring buffer the driver
    fills on its own, so a multi-second average costs no more wall clock than a
    single reading would.
    """
    output = _run_in_job(
        jobid, [NODE_PYTHON, GPU_PROBE_SCRIPT, str(GPU_SAMPLE_WINDOW)]
    )
    if output is None:
        return None

    gpus = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            card = json.loads(line)
        except ValueError as exc:
            print(f"Warning: unreadable gpu_probe line {line[:80]!r} for job {jobid}: {exc}")
            continue
        gpus.append(
            {
                "index": card["index"],
                "name": re.sub(r"^NVIDIA\s+", "", card["name"] or "GPU"),
                "util": card["util_avg"],
                "util_window_s": card["util_window_s"],
                "util_n": card["util_n"],
                "mem_used_mb": card["mem_used_mb"],
                "mem_total_mb": card["mem_total_mb"],
                "power": card["power_avg_w"],
                "power_window_s": card["power_window_s"],
                "power_n": card["power_n"],
                "power_limit": card["power_limit_w"],
                "temp": card["temp_c"],
                "source": "nvml",
            }
        )
    if not gpus:
        print(f"Warning: gpu_probe of job {jobid} returned no GPU, ignoring it")
        return None
    return gpus


def _probe_gpu_smi(jobid):
    """Fall back to nvidia-smi, whose readings are a single instant, not a mean.

    Kept for a node where NVML cannot be reached the way gpu_probe.py does it;
    the rows it produces are marked so the page does not claim an average it
    never took.
    """
    output = _run_in_job(
        jobid,
        [NVIDIA_SMI, f"--query-gpu={NVIDIA_SMI_FIELDS}", "--format=csv,noheader,nounits"],
    )
    if output is None:
        return None

    gpus = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            print(f"Warning: unexpected nvidia-smi row {line!r} for job {jobid}, skipping")
            continue
        index, name, util, mem_used, mem_total, power, power_limit, temp = parts
        gpus.append(
            {
                "index": index,
                "name": re.sub(r"^NVIDIA\s+", "", name),
                "util": _smi_number(util, jobid, "utilization.gpu"),
                "util_window_s": None,
                "util_n": None,
                "mem_used_mb": _smi_number(mem_used, jobid, "memory.used"),
                "mem_total_mb": _smi_number(mem_total, jobid, "memory.total"),
                "power": _smi_number(power, jobid, "power.draw"),
                "power_window_s": None,
                "power_n": None,
                "power_limit": _smi_number(power_limit, jobid, "power.limit"),
                "temp": _smi_number(temp, jobid, "temperature.gpu"),
                "source": "nvidia-smi",
            }
        )
    if not gpus:
        print(f"Warning: nvidia-smi probe of job {jobid} returned no GPU, ignoring it")
        return None
    return gpus


def _probe_gpu(jobid):
    """One job's GPUs, preferring the averaged NVML reading over a point one."""
    return _probe_gpu_nvml(jobid) or _probe_gpu_smi(jobid)


def _probe_gpus(jobids):
    """Probe several jobs at once, keyed by job id; jobs that failed are absent."""
    if not jobids:
        return {}
    with ThreadPoolExecutor(max_workers=min(GPU_PROBE_WORKERS, len(jobids))) as pool:
        probed = list(pool.map(_probe_gpu, jobids))
    return {jobid: gpus for jobid, gpus in zip(jobids, probed) if gpus}


def _cpu_rate(jobid, cpu_seconds, elapsed_seconds, now):
    """Cores busy in this job as ``(cores, is_recent)``.

    sstat only reports the CPU time burnt so far, so the live figure is the
    difference between two refreshes divided by the time between them.  The
    first refresh after the app starts has nothing to subtract from and falls
    back to the average over the whole run, which is flagged by ``is_recent``.
    """
    previous = _CPU_SAMPLES.get(jobid)
    average = cpu_seconds / elapsed_seconds if elapsed_seconds else None

    if previous is None:
        _CPU_SAMPLES[jobid] = (cpu_seconds, now)
        return average, False

    prev_cpu, prev_at = previous
    gap = now - prev_at
    if gap < CPU_SAMPLE_MIN_GAP:
        # Two refreshes in quick succession land inside one accounting sample.
        # Keep the older reading so the next refresh still has a real baseline.
        return average, False

    _CPU_SAMPLES[jobid] = (cpu_seconds, now)
    if cpu_seconds < prev_cpu:
        print(
            f"Warning: CPU time of job {jobid} went backwards "
            f"({prev_cpu:.0f}s -> {cpu_seconds:.0f}s), showing the run average"
        )
        return average, False
    return (cpu_seconds - prev_cpu) / gap, True


def _short_duration(seconds):
    """Seconds to ``3d04h`` / ``4h12m`` / ``7m``, for the elapsed/limit column."""
    if seconds is None:
        return "&infin;"
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def _gb(megabytes):
    """Megabytes as a short GB string, keeping a decimal below 10 GB."""
    gigabytes = megabytes / 1024
    return f"{gigabytes:.1f}" if gigabytes < 10 else f"{gigabytes:.0f}"


def _tres_count(text):
    """A plain integer TRES count, or None when the field was missing."""
    if text is None or not text.isdigit():
        return None
    return int(text)


def _accounting_gpu_row(counters):
    """One GPU row rebuilt from sstat, for when both probes failed.

    SLURM samples the job's counters every JobAcctGatherFrequency seconds (30 on
    Rivanna) and keeps no power or temperature at all, so this row is both
    coarser and thinner than a probed one, and says so.
    """
    if counters is None:
        return None
    if counters["gpu_util"] is None and counters["gpu_mem_mb"] is None:
        return None
    return [
        {
            "index": None,
            "name": None,
            "util": counters["gpu_util"],
            "util_window_s": ACCOUNTING_SAMPLE_SECONDS,
            "util_n": None,
            "mem_used_mb": counters["gpu_mem_mb"],
            "mem_total_mb": None,
            "power": None,
            "power_window_s": None,
            "power_n": None,
            "power_limit": None,
            "temp": None,
            "source": "accounting",
        }
    ]


def _cpu_cell(cores, recent, alloc_cpus, span):
    """CPU column: cores busy right now against the cores the job was given."""
    if cores is None:
        return f'<td{span}><span class="queue-muted">n/a</span></td>'
    mark = "" if recent else '<span class="cpu-average">avg</span> '
    if not alloc_cpus:
        return f"<td{span}>{mark}{cores:.1f} cores</td>"
    text = f"{cores:.1f}/{alloc_cpus}"
    bar = get_resource_bar(min(cores, alloc_cpus), alloc_cpus, text)
    return f"<td{span}>{mark}{bar}</td>"


def _ram_cell(counters, alloc_mem_mb, span):
    """RAM column: resident memory now, against what the job reserved."""
    used = counters["mem_mb"]
    peak = counters["peak_mem_mb"]
    title = f' title="peak so far {_gb(peak)} GB"' if peak else ""
    if not alloc_mem_mb:
        return f"<td{span}{title}>{_gb(used)} GB</td>"
    text = f"{_gb(used)}/{_gb(alloc_mem_mb)}G"
    bar = get_resource_bar(min(used, alloc_mem_mb), alloc_mem_mb, text)
    return f"<td{span}{title}>{bar}</td>"


DASH = '<td class="dimmed">&mdash;</td>'
# What SLURM's own sampler manages, for the row that has nothing better.
ACCOUNTING_SAMPLE_SECONDS = 30


def _window_title(gpu, field):
    """Tooltip spelling out what window a reading actually averages over.

    The driver buffers roughly 14s of utilisation samples but only ~2.4s of
    power samples, so asking for five seconds gets five of the one and less of
    the other; the number shown is the span the samples really covered.
    """
    window = gpu[f"{field}_window_s"]
    if gpu["source"] == "nvidia-smi":
        return ' title="single reading, not an average: nvidia-smi fallback"'
    if gpu["source"] == "accounting":
        return f' title="from SLURM accounting, sampled every {ACCOUNTING_SAMPLE_SECONDS}s"'
    if not window:
        return ' title="no samples buffered for this window"'
    count = gpu[f"{field}_n"]
    thin = ", only 1 sample" if count == 1 else ""
    return f' title="mean of {count} samples over the last {window:.1f}s{thin}"'


def _gpu_cells(gpu, show_index):
    """The five GPU columns of one row, dashed out for a job without a GPU.

    Under cgroup isolation a one-GPU job always sees its card as index 0, so the
    index is only worth printing when the job holds more than one.
    """
    if gpu is None:
        return [DASH] * 5

    if gpu["name"]:
        prefix = f'{gpu["index"]}: ' if show_index else ""
        cells = [f'<td>{prefix}{escape(gpu["name"])}</td>']
    else:
        cells = ['<td><span class="queue-muted">(accounting)</span></td>']
    if gpu["source"] == "nvidia-smi":
        cells[0] = cells[0].replace(
            "</td>", ' <span class="queue-muted">(instant)</span></td>'
        )

    if gpu["util"] is None:
        cells.append(DASH)
    else:
        bar = get_resource_bar(gpu["util"], 100, f'{gpu["util"]:.0f}%')
        cells.append(f'<td{_window_title(gpu, "util")}>{bar}</td>')

    used, total = gpu["mem_used_mb"], gpu["mem_total_mb"]
    if used is None:
        cells.append(DASH)
    elif total:
        bar = get_resource_bar(used, total, f"{_gb(used)}/{_gb(total)}G")
        cells.append(f"<td>{bar}</td>")
    else:
        cells.append(f"<td>{_gb(used)} GB</td>")

    draw, limit = gpu["power"], gpu["power_limit"]
    title = _window_title(gpu, "power")
    if draw is None:
        cells.append(DASH)
    elif limit:
        bar = get_resource_bar(min(draw, limit), limit, f"{draw:.0f}/{limit:.0f}W")
        cells.append(f"<td{title}>{bar}</td>")
    else:
        cells.append(f"<td{title}>{draw:.0f} W</td>")

    if gpu["temp"] is None:
        cells.append(DASH)
    else:
        cells.append(f'<td class="{_temp_css(gpu["temp"])}">{gpu["temp"]:.0f} &deg;C</td>')
    return cells


def _temp_css(celsius):
    """Colour a GPU temperature: A40/A6000 boards throttle in the high eighties."""
    if celsius >= 85:
        return "queue-pressure-high"
    if celsius >= 75:
        return "queue-pressure-mid"
    return "queue-pressure-low"


def parse_my_jobs_to_table():
    """Live CPU, RAM and GPU readings for the running jobs of this account.

    Two sources are combined per refresh.  One sstat call covers every job at
    once and gives the CPU time and resident memory the accounting plugin last
    sampled; then one short nvidia-smi step per GPU job gives the readings SLURM
    does not keep, above all the power draw and the board temperature.  A job
    whose probe fails still shows the GPU utilisation and memory that sstat
    collected, marked as coming from accounting rather than from the card.
    """
    owner = _app_owner()
    jobs = _my_running_jobs()
    if not jobs:
        return (
            f'<p class="queue-note">{escape(display_name(owner))} has no running '
            "job right now.</p>"
        )

    jobids = [job["jobid"] for job in jobs]
    usage = _sstat_usage(jobids)
    gpu_jobs = [job["jobid"] for job in jobs if _alloc_gpu_count(job["tres-alloc"])]
    probes = _probe_gpus(gpu_jobs)

    # Jobs that have ended since the last refresh must not keep a sample around,
    # or a recycled job id would be handed someone else's counter.
    for stale in set(_CPU_SAMPLES) - set(jobids):
        del _CPU_SAMPLES[stale]

    now = time.time()
    columns = ("JOB", "PARTITION", "ELAPSED", "CPU", "RAM",
               "GPU", "UTIL", "GPU MEM", "POWER", "TEMP")
    html = ["<table>", "<tr>"]
    html += [f"<th>{column}</th>" for column in columns]
    html.append("</tr>")

    stale_cpu = False
    accounting_only = False
    instant_only = False
    for job in jobs:
        jobid = job["jobid"]
        tres = job["tres-alloc"]
        counters = usage.get(jobid)
        alloc_cpus = _tres_count(_tres_field(tres, "cpu"))
        alloc_mem_mb = _mem_to_mb(_tres_field(tres, "mem") or "")
        elapsed = _slurm_time_to_seconds(job["timeused"])
        limit = _slurm_time_to_seconds(job["timelimit"])

        # One row per GPU, so a multi-GPU job shows each card on its own line.
        gpus = probes.get(jobid)
        if gpus is None:
            gpus = _accounting_gpu_row(counters)
            accounting_only = accounting_only or bool(gpus)
        elif any(gpu["source"] == "nvidia-smi" for gpu in gpus):
            instant_only = True
        rows = gpus or [None]
        span = f' rowspan="{len(rows)}"' if len(rows) > 1 else ""

        for position, gpu in enumerate(rows):
            html.append("<tr>")
            if position == 0:
                html.append(
                    f'<td{span}>{jobid}<br><span class="job-name">'
                    f'{escape(job["name"])}</span></td>'
                )
                html.append(f'<td{span}>{escape(job["partition"])}</td>')
                html.append(
                    f"<td{span}>{_short_duration(elapsed)}"
                    f'<span class="queue-muted"> / {_short_duration(limit)}</span></td>'
                )

                if counters is None:
                    # Between the job starting and its first step registering,
                    # sstat has nothing to report for it yet.
                    pending = '<td{0}><span class="queue-muted">starting…</span></td>'
                    html.append(pending.format(span))
                    html.append(pending.format(span))
                else:
                    cores, recent = _cpu_rate(
                        jobid, counters["cpu_seconds"], elapsed, now
                    )
                    stale_cpu = stale_cpu or not recent
                    html.append(_cpu_cell(cores, recent, alloc_cpus, span))
                    html.append(_ram_cell(counters, alloc_mem_mb, span))

            html += _gpu_cells(gpu, len(rows) > 1)
            html.append("</tr>")

    html.append("</table>")

    notes = [
        f"Readings for <b>{escape(display_name(owner))}</b>, taken when this "
        "section was last refreshed. CPU and RAM come from SLURM's accounting "
        "sampler; the GPU columns are read off the card by a short step inside "
        "each job."
    ]
    notes.append(
        f"<b>UTIL</b> is the mean over the last ~{GPU_SAMPLE_WINDOW:.0f}s and "
        "<b>POWER</b> over the last ~2.4s, taken from samples the driver has "
        "already buffered, so averaging them costs no extra wait. A single "
        "instant of utilisation swings tens of points on a steady job, which is "
        "why it is not shown raw. <b>GPU MEM</b> and <b>TEMP</b> are point "
        "readings, being levels rather than rates. Hover a bar for the window "
        "and sample count behind it."
    )
    if stale_cpu:
        notes.append(
            "A CPU figure marked <span class=\"cpu-average\">avg</span> is the "
            "average over the whole run: the live number needs two refreshes "
            f"at least {CPU_SAMPLE_MIN_GAP}s apart to subtract."
        )
    if instant_only:
        notes.append(
            "A GPU row marked <span class=\"queue-muted\">(instant)</span> fell "
            "back to <code>nvidia-smi</code>, which reports one moment rather "
            "than an average, so its utilisation is the noisy figure."
        )
    if accounting_only:
        notes.append(
            "A GPU row marked <span class=\"queue-muted\">(accounting)</span> "
            "could not be read on the card at all, so it falls back to what "
            f"SLURM sampled every {ACCOUNTING_SAMPLE_SECONDS}s; power and "
            "temperature are not collected there."
        )
    html += [f'<p class="queue-note">{note}</p>' for note in notes]
    return "".join(html)

def slurm_response(build, *args, **kwargs):
    """Render one panel, degrading to a note when the query cannot be made.

    parse_cmd already retries both a transient controller error and a login node
    too short of memory to fork; if it still fails the panel says which of the
    two it was and the next refresh picks it up again, which beats a 500 and a
    traceback in the browser.
    """
    try:
        return Response(build(*args, **kwargs), mimetype="text")
    except CalledProcessError as exc:
        print(f"Warning: {build.__name__} gave up, SLURM command failed: {exc}")
        note = (
            "SLURM did not answer just now &mdash; the controller is usually only "
            "busy for a few seconds, so the next refresh should fill this back in."
        )
    except OSError as exc:
        print(f"Warning: {build.__name__} gave up, could not run a SLURM command: {exc}")
        note = (
            "This login node could not start the SLURM commands just now "
            f"(<code>{exc}</code>). It is out of memory rather than out of GPUs; "
            "try again in a moment, or from a less loaded login node."
        )
    return Response(f'<p class="queue-note">{note}</p>', mimetype="text", status=503)


def main():
    parser = argparse.ArgumentParser(description="launch web app")
    parser.add_argument(
        "--host", default="0.0.0.0", help="the host address for the website"
    )
    parser.add_argument(
        "--port", default=2070, type=int, help="the port for the website"
    )
    args = parser.parse_args()

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(
            open("index.html").read(),
            hostname=args.host,
            accounts=LAB_ACCOUNTS,
            my_jobs_min_gap_ms=CPU_SAMPLE_MIN_GAP * 1000,
        )

    @app.route("/time_feed")
    def time_feed():
        def generate():
            yield f'updated at: {datetime.now(pytz.timezone("America/New_York")).strftime("%Y.%m.%d | %H:%M:%S")}'

        return Response(generate(), mimetype="text")

    @app.route("/resource")
    def resource():
        return slurm_response(parse_usage_to_table)

    @app.route("/queue")
    def queue():
        return slurm_response(parse_queue_to_table, selected_account(request.args))

    @app.route("/queue_stats")
    def queue_stats():
        return slurm_response(parse_queue_stats_to_table)

    @app.route("/wait_estimate")
    def wait_estimate():
        try:
            job = parse_standard_job_args(request.args)
        except InvalidJobRequest as err:
            print(f"Warning: rejected standard-job request {dict(request.args)}: {err}")
            return Response(
                f'<p class="queue-note">Cannot run that job shape: {err}</p>',
                mimetype="text",
                status=400,
            )

        return slurm_response(parse_wait_estimate_to_table, job)

    @app.route("/priority")
    def priority():
        return slurm_response(parse_priority_to_table, selected_account(request.args))

    @app.route("/leaderboard")
    def leaderboard():
        return slurm_response(parse_leaderboard)

    @app.route("/leaderboard_partition")
    def leaderboard_partition():
        return slurm_response(parse_leaderboard_by_partition)

    @app.route("/disk_quota")
    def disk_quota():
        return slurm_response(parse_disk_quota)

    @app.route("/allocations")
    def allocations():
        return slurm_response(parse_allocations_to_table, selected_account(request.args))

    @app.route("/my_jobs")
    def my_jobs():
        return slurm_response(parse_my_jobs_to_table)

    # @app.route('/cpu_resource')
    # def cpu_resource():
    #     def generate():
    #         out = parse_cpu_usage_to_table()
    #         yield out
    #     return Response(generate(), mimetype='text')

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
