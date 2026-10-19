import os
import re
from datetime import datetime, timedelta
import argparse
import functools
import json
import pwd
import socket
import sys
import threading
import bisect
import copy
import time
# Before numpy: importing slurm_gpustat pins the BLAS thread pools, which only
# takes effect if numpy has not been loaded yet.
from slurm_gpustat import (
    resource_by_type,
    parse_all_gpus,
    gpu_usage,
    node_states,
    is_down,
    is_inaccessible,
    split_by_access,
    partition_nodes,
    NONAME_GPU,
    parse_cmd,
    avail_stats_for_node,
    node_records,
    parse_node_names,
    squeue_fields,
    squeue_jobs,
    parse_gres_gpus,
    tres_field,
    mem_to_mb,
    alloc_gpu_count,
    reserved_for_others,
    aggregate_usage,
    subtract_usage,
    format_size,
    ttl_cache,
    CMD_CACHE_SECONDS,
    MONITORED_PARTITIONS,
    LAB_ACCOUNTS,
)
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

# Every GPU type the page knows: its compute capability
# (https://developer.nvidia.com/cuda-gpus), which orders types most capable first,
# and the memory label shown next to it. A type missing here sorts first and goes
# unlabelled, so add a row when a new one shows up.
GPU_TYPES = {
    "1g.10gb": (8.0, "[10g]"),
    "1g.24gb": (8.0, "[24g]"),
    "b200": (10.0, "[192g]"),
    "rtx_pro_6000": (12.0, "[96g]"),
    "h200": (8.9, "[141g]"),
    "a4500": (8.0, "[20g]"),
    "a100": (8.0, "[40/80g]"),
    "a6000": (8.6, "[48g]"),
    "a40": (8.6, "[48g]"),
    "a30": (8.0, "[24g]"),
    "a10": (8.6, ""),
    "a16": (8.6, ""),
    "rtx_3090": (8.6, "[24g]"),
    "rtx_2080": (7.5, "[11g]"),
    "v100": (7.0, "[16g]"),
    "gv100gl": (7.0, "[32g]"),
    "v100s": (7.0, "[32g]"),
    "p40": (6.1, "[24g]"),
    "m40": (5.2, "[12/24g]"),
    "rtx6k": (7.5, "[24g]"),
    "rtx8k": (7.5, "[48g]"),
    "mig": (None, "[11g]"),
}


def _type_rank(gpu_type):
    """Sort key putting the most capable GPU first, ties broken by name.

    Several types share a capability number (a40, a6000 and rtx_3090 are all
    8.6), and the lists being sorted are built from sets, whose order follows
    string hashing and so is randomised per process. Without the name the tied
    types came out in a different order every time the app was restarted.
    """
    capability = GPU_TYPES.get(gpu_type, (None, ""))[0]
    return (-(10.0 if capability is None else capability), gpu_type)


def _mem_label(gpu_type):
    """The memory label shown next to a GPU type, e.g. "[48g]"; "" if unknown."""
    return GPU_TYPES.get(gpu_type, (None, ""))[1]


OLD_GPU_TYPES = ["p40", "m40"]
# Memory sizes the leaderboard also totals on their own, as "48g=N".
LEADERBOARD_GMEM = (48,)


def get_resource_bar(avail, total, text=None, long=False):
    """Create a long/short progress bar with text overlaid. Formatting handled in css.

    The text defaults to "avail / total".
    """
    if text is None:
        text = f"{avail} / {total}"
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


def _leaderboard_row(user, subdict, name_width, owner):
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
    for gm in LEADERBOARD_GMEM:
        total += f"|{gm}g={str(sum([val for key, val in subdict['n_gpu'].items() if _mem_label(key) == f'[{gm}g]'])):2s}"
    total += f"|newer={str(sum(num_new_gpus)):2s}"
    total += f"|bash={str(sum(subdict['bash_gpu'].values())):2s}"
    line = escape(f"{display_name(user):{name_width}s}[{total}]    {summary_str}")
    if user == owner:
        return f'<span class="leaderboard-me">&gt; {line}</span>\n'
    return f"  {line}\n"


def parse_leaderboard():
    """GPUs held by each user across the monitored partitions, most GPUs first."""
    aggregates = aggregate_usage(gpu_usage(resources=parse_all_gpus()))
    owner = _app_owner()
    name_width = max((len(display_name(u)) for u in aggregates), default=12) + 2
    ranked = sorted(aggregates.items(), key=lambda x: sum(x[1]["n_gpu"].values()), reverse=True)
    return "".join(_leaderboard_row(user, subdict, name_width, owner) for user, subdict in ranked)


def _state_badge(state):
    """Return an HTML badge for the node state."""
    s = state.lower()
    if is_down(state):
        cls = "state-down"
    elif is_inaccessible(state):
        cls = "state-resv"
    elif "mix" in s or "alloc" in s:
        cls = "state-busy"
    else:
        cls = "state-idle"
    return f'<span class="state-badge {cls}">{state}</span>'


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


def _is_mig_type(gpu_type):
    """MIG slice types are named after their memory share, e.g. "1g.10gb"."""
    return gpu_type[0].isdigit() if gpu_type else False


def _split_mig(items, type_of=lambda item: item):
    """``items`` as (full GPUs, MIG slices), each keeping its order."""
    return (
        [item for item in items if not _is_mig_type(type_of(item))],
        [item for item in items if _is_mig_type(type_of(item))],
    )


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
        node: name for node, name in reserved_for_others().items() if node in resources
    }
    # Reserved for somebody else is as good as down for us, whatever sinfo says.
    res_accessible, res_down, excluded = split_by_access(resources, states, reserved)
    for node in reserved:
        states[node] = "resv"

    res_total = copy.deepcopy(res_accessible)
    usage = gpu_usage(resources=res_accessible, excluded=excluded)
    subtract_usage(res_accessible, usage)

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


def parse_usage_to_table():
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

    gpu_type_list, mig_type_list = _split_mig(sorted(all_gpu_types, key=_type_rank))

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
                gpu_bar = get_resource_bar(gpu_count_avail.get(node, 0), gpu_count_total[node])
                gpu_stat = f"<td>gpu: {gpu_bar}</td>"

                users_list = [user for user in usage if node in usage[user].get(gpu_type, [])]
                if len(users_list):
                    users_td = f"<td>user: {','.join(users_list)}</td>"
                else:
                    users_td = "<td>&nbsp</td>"

                stats = avail_stats_for_node(node)
                cpu_free, cpu_total = stats["cpu"]
                cpu_bar = get_resource_bar(cpu_free, cpu_total)
                mem_free, mem_total = stats["mem"]
                mem_bar = get_resource_bar(
                    mem_free // 1024**3, mem_total // 1024**3,
                    text=f"{format_size(mem_free)} / {format_size(mem_total)}", long=True,
                )
                cpu_stat = f"<td>cpu: {cpu_bar}</td>"
                mem_stat = f"<td>mem: {mem_bar}</td>"

                node_summaries.append(f"<tr><td>&nbsp</td>{node_name_td}{state_html}{gpu_stat}{cpu_stat}{mem_stat}{users_td}</tr>")

            for node in down_node_names:
                node_state = states.get(node, "unknown")
                held_by = f' <span class="queue-muted">{reserved[node]}</span>' if node in reserved else ""
                state_html = f"<td>{_state_badge(node_state)}{held_by}</td>"
                node_name_td = f"<td>{node}</td>"
                node_gpu_count = sum(x["count"] for x in res_down[node])
                down_summaries.append(
                    f'<tr class="row-down"><td>&nbsp</td>{node_name_td}{state_html}'
                    f'<td class="dimmed">gpu: 0 / {node_gpu_count}</td>'
                    f'<td class="dimmed">-</td><td class="dimmed">-</td><td>&nbsp</td></tr>'
                )

            type_total = sum(gpu_count_total.values())
            type_avail = sum(gpu_count_avail.values())
            type_bar = get_resource_bar(type_avail, type_total)

            down_note = _down_note(down_node_dicts)

            rows_html.append(
                f'<tr><td colspan="7"><b>{gpu_type} {_mem_label(gpu_type)}: {type_bar} gpus available{down_note}</b></td></tr>'
            )
            rows_html.extend(node_summaries)
            rows_html.extend(down_summaries)
            total_count += type_total
            avail_count += type_avail

        down_entries = [x for t in type_list_to_render for x in res_down_by_type.get(t, [])]
        return "".join(rows_html), avail_count, total_count, down_entries

    gpu_rows, gpu_avail, gpu_total, gpu_down = _render_type_rows(gpu_type_list)
    mig_rows, mig_avail, mig_total, mig_down = _render_type_rows(mig_type_list)

    total_bar = get_resource_bar(gpu_avail, gpu_total)
    down_note = _down_note(gpu_down)
    total_summary = f'<tr><td colspan="7"><h3>Summary: {total_bar} gpus available{down_note}</h3></td></tr>'

    table_html = f"<table>{total_summary}{gpu_rows}</table>"
    table_html += (
        '<p class="queue-note">Available means unallocated in Slurm. A job also needs '
        "enough CPUs and memory on the same node, plus a scheduling window for its "
        "walltime. These counts alone do not mean it can start now.</p>"
    )

    if mig_type_list:
        mig_bar = get_resource_bar(mig_avail, mig_total)
        mig_down_note = _down_note(mig_down)
        table_html += _folded(f"MIG: {mig_bar} slices available{mig_down_note}", mig_rows)

    return table_html


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


# What squeue prints for a job with no node list or constraint.
UNSET_JOB_FIELD = ("", "(null)", "N/A")
# The --constraint syntax read here: features joined by & (and) or | (or). Matching-OR
# brackets around the whole expression are dropped, since for whether one node
# qualifies they are a plain or. Brackets inside it (a100&[a40|a6000]), counts
# (a100*2), parentheses and negation are not read, and leave the job free to use
# any node.
SIMPLE_CONSTRAINT = re.compile(r"[\w.-]+(?:[&|][\w.-]+)*")
MATCHING_OR = re.compile(r"\[([^\[\]]+)\]")


class _NodeEligibility:
    """Which nodes a pending job may use, from its -w, -x and --constraint.

    Built once per panel refresh from ``members`` (partition -> nodes): node
    features come from one scontrol table, and each distinct constraint, and each
    distinct request, a few dozen against thousands of jobs, is worked out once. A
    node with no scontrol record (the table could not be read) counts as having
    every feature, so a failed read leaves jobs unfiltered rather than turning them
    all away.
    """

    def __init__(self, records, members):
        self.members = members
        self.nodes = set().union(*members.values())
        self.features = {
            node: set(records[node].get("ActiveFeatures", "").split(","))
            for node in self.nodes if node in records
        }
        missing = len(self.nodes) - len(self.features)
        if missing:
            print(f"Warning: no scontrol record for {missing} of {len(self.nodes)} nodes, "
                  "so job constraints cannot rule them out")
        self.by_constraint = {}
        self.by_request = {}
        self.unread = set()

    def _matching(self, expression):
        """Nodes satisfying ``expression``, every node when it cannot be read."""
        if expression not in self.by_constraint:
            wrapped = MATCHING_OR.fullmatch(expression)
            simple = wrapped.group(1) if wrapped else expression
            if SIMPLE_CONSTRAINT.fullmatch(simple):
                alternatives = [set(option.split("&")) for option in simple.split("|")]
                self.by_constraint[expression] = {
                    node for node in self.nodes
                    if node not in self.features
                    or any(option <= self.features[node] for option in alternatives)
                }
            else:
                self.unread.add(expression)
                self.by_constraint[expression] = self.nodes
        return self.by_constraint[expression]

    def job_nodes(self, job, partitions):
        """The nodes of ``partitions`` that ``job`` may run on, as a frozenset shared
        with every job that asks the same way."""
        key = (tuple(partitions), job["excnodes"], job["feature"], job["reqnodes"],
               job["numnodes"])
        if key not in self.by_request:
            self.by_request[key] = frozenset(self._eligible(*key))
        return self.by_request[key]

    def _eligible(self, partitions, excluded, expression, required, numnodes):
        candidates = set().union(*(self.members.get(p, ()) for p in partitions))
        if excluded not in UNSET_JOB_FIELD:
            candidates -= set(parse_node_names(excluded))
        if expression not in UNSET_JOB_FIELD:
            candidates &= self._matching(expression)
        if required not in UNSET_JOB_FIELD:
            # -w names nodes the job must all get, not a choice among them: if one is
            # unusable here, so is the job. With more nodes asked for than listed it
            # also takes others, and those may be any candidate; the wait replay
            # does not force the listed ones into its pick (no live job does this).
            required = set(parse_node_names(required))
            if not required <= candidates:
                return set()
            if len(required) >= (_first_int(numnodes) or 1):
                return required
        return candidates

    def report(self, panel):
        """Say once which constraints were not read, rather than once per job."""
        if self.unread:
            print(f"Warning: {panel} could not read {len(self.unread)} job constraints and "
                  f"let those jobs use any node: {', '.join(sorted(self.unread))}")


def parse_queue_stats_to_table():
    """Aggregate the pending GPU jobs of the monitored partitions by the GPU type
    they are actually waiting for.

    A pending job may list several partitions and may not name a GPU type at all, so
    the types it can land on are those of the nodes in its partitions that its -w, -x
    and --constraint leave it; a job left none counts as blocked. Jobs left
    with a single candidate type are committed demand for that type; jobs with several
    candidates are reported separately, because they will only ever consume one of
    them. Jobs are also split by their pending reason, so that the queue depth that
    actually competes for GPUs is not drowned out by jobs held back by dependencies.
    """
    # Only MONITORED_PARTITIONS: the GPU counts demand is weighed against come from
    # them, so a job queued elsewhere (bii-gpu) is not demand for those GPUs.
    members = partition_nodes()
    inventory = parse_all_gpus(partition=MONITORED_PARTITIONS)

    @functools.cache
    def _types_on(nodes):
        """GPU types on a frozenset of nodes; jobs asking alike share one set."""
        return frozenset(entry["type"] for node in nodes for entry in inventory.get(node, ()))

    partition_types = {p: types for p, nodes in members.items()
                       if (types := _types_on(frozenset(nodes)))}
    eligibility = _NodeEligibility(node_records(), members)

    def _new_bucket():
        return {category: {"jobs": 0, "gpus": 0} for category in QUEUE_CATEGORIES}

    committed = defaultdict(_new_bucket)  # gpu type -> category -> counts
    flexible = defaultdict(_new_bucket)  # gpu type -> category -> counts
    totals = _new_bucket()
    # {partition_group: {"jobs": N, "gpus": N, "types": {type: count}}}
    group_stats = defaultdict(lambda: {"jobs": 0, "gpus": 0, "types": defaultdict(int)})
    impossible_requests = set()
    # Routine skips, tallied and reported once below.
    not_gpu_partition = no_gpu_request = 0

    for job in squeue_jobs("PENDING"):
        partition_list = [p.strip() for p in job["partition"].split(",")]
        # The monitored partitions whose nodes have GPUs.
        gpu_parts = [p for p in partition_list if p in partition_types]
        if not gpu_parts:
            not_gpu_partition += 1
            continue

        # tres-alloc holds the whole job's GPUs however they were asked for
        # (--gres=gpu, a bare one included, --gpus, --gpus-per-task); only the
        # tres-per-* fields say which type, if any, was named.
        job_gpus = alloc_gpu_count(job["tres-alloc"])
        if not job_gpus:
            no_gpu_request += 1
            continue

        req_type = _gres_gpu_type(job)
        # A pending array's tasks share one row, e.g. tasks 0-5,8%2
        tasks = job["arraytaskid"]
        num_tasks = _array_task_count(tasks)
        total_gpus = job_gpus * num_tasks

        group_key = ", ".join(sorted(gpu_parts))
        candidates = _types_on(eligibility.job_nodes(job, gpu_parts))
        if req_type:
            if not any(req_type in partition_types[p] for p in gpu_parts):
                # The named type is offered by none of the requested partitions; keep
                # the request visible instead of silently dropping the job.
                impossible_requests.add((group_key, req_type))
            # Its node lists or constraint may rule out every node of that type even
            # where the partitions offer it; either way it can never start.
            runnable = req_type in candidates
            candidates = {req_type}
        else:
            runnable = bool(candidates)

        category = _classify_pending_reason(job["reason"]) if runnable else "blocked"
        totals[category]["jobs"] += num_tasks
        totals[category]["gpus"] += total_gpus

        target = committed if len(candidates) == 1 else flexible
        for gpu_type in candidates:
            target[gpu_type][category]["jobs"] += num_tasks
            target[gpu_type][category]["gpus"] += total_gpus

        group_stats[group_key]["jobs"] += num_tasks
        group_stats[group_key]["gpus"] += total_gpus
        group_stats[group_key]["types"][req_type if req_type else "any"] += total_gpus

    if not_gpu_partition or no_gpu_request:
        print(f"Note: queue overview skipped {not_gpu_partition} pending jobs outside the "
              f"monitored GPU partitions and {no_gpu_request} that ask for no GPU")
    eligibility.report("queue overview")
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
    gpu_type_list, mig_type_list = _split_mig(sorted(all_types, key=_type_rank))

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
            throttled_gpus = own["throttled"]["gpus"] + flex["throttled"]["gpus"]
            blocked_gpus = own["blocked"]["gpus"] + flex["blocked"]["gpus"]

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
                f"<tr><td><b>{gpu_type}</b> {_mem_label(gpu_type)}</td>"
                f"<td>{get_resource_bar(free, capacity)}</td>"
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
        f"&nbsp;|&nbsp; blocked by dependencies, holds or no usable node: "
        f'{totals["blocked"]["jobs"]} jobs / {totals["blocked"]["gpus"]} gpus</span>'
        "</h3></td></tr>"
    )

    table_html = f"<table>{summary}{header}{_render_type_rows(gpu_type_list)}</table>"
    table_html += (
        '<p class="queue-note">A job that names several partitions is counted under '
        "every GPU type it could land on, so <b>Flexible</b> and <b>Not Competing</b> overlap between "
        "rows; only the summary line above is a plain per-job total.</p>"
    )

    if mig_type_list:
        table_html += _folded("MIG slices", header + _render_type_rows(mig_type_list))

    groups = sorted(group_stats.items(), key=lambda x: x[1]["gpus"], reverse=True)
    group_rows = ""
    for group, data in groups:
        type_parts = []
        for gpu_type, count in sorted(data["types"].items(), key=lambda x: x[1], reverse=True):
            if gpu_type == "any":
                type_parts.append(f'<span class="gpu-type-any">{gpu_type}&times;{count}</span>')
            else:
                type_parts.append(f"{gpu_type}&times;{count}")
        group_rows += _table_row([group, data["jobs"], data["gpus"], ", ".join(type_parts)])
    all_jobs = sum(v["jobs"] for v in totals.values())
    all_gpus = sum(v["gpus"] for v in totals.values())
    table_html += _folded(
        f"Raw breakdown by requested partitions ({all_jobs} jobs, {all_gpus} gpus)",
        _table_row(["Partition(s)", "Pending Jobs", "GPUs Needed", "Requested GPU Type"], "th")
        + group_rows
        + f'<tr class="queue-stats-total"><td><b>Total</b></td><td><b>{all_jobs}</b></td>'
        f"<td><b>{all_gpus}</b></td><td></td></tr>",
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
_reported_no_access_guard = threading.Lock()
# Whether an account may use a pool is account, QOS and partition configuration,
# which barely moves between refreshes, yet the panel asked slurmctld to simulate
# the same fourteen jobs on every one.  One marker above does follow live state:
# "Requested node configuration is not available" also fires while every node of
# a type is drained, so the verdict is kept briefly rather than for the life of
# the process, and a pool that comes back is picked up within the TTL.
TEST_ONLY_CACHE_SECONDS = int(os.environ.get("RIVANNA_TEST_ONLY_TTL", "60"))
# Threads the wait estimate runs its SLURM reads and test-only checks on.
WAIT_ESTIMATE_WORKERS = 4
# Union partitions already reported as skipped, so that log is not repeated either.
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
        for gpu_type, n_gpu in parse_gres_gpus(tokens[3]):
            gpu_type = gpu_type or NONAME_GPU
            if n_gpu is None:
                print(f"Warning: partition {partition} gives no {gpu_type} gpu count, skipping")
                continue
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
                job_mem = max(mem // n_gpu, 1024) * job["gpus"]
            else:
                job_mem = job["mem_gb"] * 1024

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

    Only the admission verdict is used; the separate replay produces an
    approximate start under its stated assumptions. Returns None when the job is admissible, else a
    ``("no_access" | "error", reason)`` tuple.
    """
    return _test_only_verdict(
        account, spec["partition"], spec["gpu_type"],
        spec["gpus"], spec["cpus"], spec["mem_mb"], spec["minutes"],
    )


@ttl_cache(TEST_ONLY_CACHE_SECONDS)
def _test_only_verdict(account, partition, gpu_type, gpus, cpus, mem_mb, minutes):
    """The cached ``sbatch --test-only`` behind :func:`_test_only_check`."""
    # Every interpolated field is validated in parse_standard_job_args or read
    # straight out of sinfo, so none of them can carry shell syntax.
    cmd = (
        f"sbatch --test-only -A {account} -p {partition} "
        f"--gres=gpu:{gpu_type}:{gpus} -c {cpus} "
        f"--mem={mem_mb}M -t {minutes} --wrap=hostname"
    )
    try:
        out = check_output(cmd, shell=True, stderr=STDOUT).decode("utf-8")
    except CalledProcessError as exc:
        out = exc.output.decode("utf-8")
    if re.search(r"to start at (\S+)", out):
        verdict = None
    else:
        lines = out.strip().splitlines()
        reason = lines[-1].replace("sbatch: error: ", "") if lines else "no output"
        pool = (partition, gpu_type)
        if any(marker in reason for marker in NO_ACCESS_MARKERS):
            with _reported_no_access_guard:
                first_report = pool not in _reported_no_access
                _reported_no_access.add(pool)
            if first_report:
                print(
                    f"Note: {account} has no schedulable node for {gpu_type} in "
                    f"{partition} ({reason}); leaving that pool out of the estimates"
                )
            verdict = ("no_access", reason)
        else:
            print(f"Warning: {gpu_type} on {partition} rejects this job shape: {reason}")
            verdict = ("error", reason)
    return verdict


# ------------------------------------------------------------ queue snapshot
# Arrays with more pending tasks than this are cut down, with a warning: the
# simulation would otherwise spend its time on a queue that cannot run anyway.
ARRAY_TASK_CAP = 2000
# Resolution and reach of the queue replay, see _plan_pool.
PLAN_MINUTES = 10
PLAN_HORIZON_DAYS = 30


def _gres_gpu_type(job):
    """GPU type an squeue row names in any of its tres-per-* fields, or None."""
    for field in ("tres-per-node", "tres-per-job", "tres-per-task"):
        match = re.search(r"gpu:([^:,()]+):\d+", job[field])
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


def _array_task_count(tasks):
    """Tasks behind one pending squeue row: its arraytaskid's count, 1 for a plain job."""
    return 1 if tasks in ("", "N/A") else _expand_array_tasks(tasks)[0]


# How much of an array's task list a JOBID cell shows: what squeue's default
# output fits before it ends the list with "...".
ARRAY_TASKS_SHOWN = 63


def _short_tasks(tasks):
    """An array's task list cut at a comma to ARRAY_TASKS_SHOWN characters and
    ended with ``...``, throttle kept. For display only; counts use the full list."""
    spec, sep, throttle = tasks.partition("%")
    if len(spec) <= ARRAY_TASKS_SHOWN:
        return tasks
    cut = spec.rfind(",", 0, ARRAY_TASKS_SHOWN - 3)
    return spec[: cut + 1] + "..." + sep + throttle


def _sprio_rows():
    """``(jobid, partition, total, partition factor)`` strings, one per sprio row.

    One query feeds both the wait estimate and the priority panel, so a refresh
    asks sprio once. Each caller checks the numbers it uses.
    """
    rows = []
    for row in parse_cmd("sprio -h -o '%i|%r|%Y|%P'"):
        fields = [f.strip() for f in row.split("|")]
        if len(fields) < 4:
            print(f"Warning: unexpected sprio row {row!r}, skipping")
            continue
        rows.append(tuple(fields[:4]))
    return rows


def _sprio_by_job():
    """jobid -> {partition: total priority} for every job SLURM is ranking.

    Multi-partition jobs get one row per partition, each with the priority the
    job has *there*. Jobs held by a dependency have no priority and are absent,
    which is exactly the set of jobs not competing for a node right now.
    """
    priorities = defaultdict(dict)
    for jobid, partition, total, _ in _sprio_rows():
        if not total.isdigit():
            print(f"Warning: unparsable sprio priority {total!r} for job {jobid}, skipping")
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


def _row_alloc(row, max_minutes, what):
    """Whole-job GPUs, CPUs, memory (MB) and time limit of one squeue row.

    ``what`` is how warnings name the job ("job", "pending job"). The time limit
    falls back to ``max_minutes`` when squeue gives none.
    """
    tres = row["tres-alloc"]
    cpus = tres_field(tres, "cpu")
    mem = tres_field(tres, "mem")
    mem_mb = mem_to_mb(mem) if mem else 0
    if mem and mem_mb is None:
        print(f"Warning: unparsable memory {mem!r} for {what} {row['jobid']}, counting 0")
        mem_mb = 0
    limit = _slurm_time_to_minutes(row["timelimit"])
    return {
        "gpus": alloc_gpu_count(tres),
        "cpus": int(cpus) if cpus and cpus.isdigit() else 0,
        "mem_mb": mem_mb,
        "minutes": max_minutes if limit is None else limit,
        "array": row["arrayjobid"] if row["arrayjobid"] not in ("", "N/A") else None,
    }


def _running_jobs_by_node(rows, now, max_minutes, wanted):
    """node -> list of running allocations on it, from squeue rows.

    Each allocation carries what the job holds on *that* node (a multi-node job
    is split evenly), when SLURM will kill it, and how much of its limit it has
    already used so the Historical Runtime estimate can shorten it. A job with no time
    limit has no end (None): nothing says when it lets go of its share. Only
    jobs on a node in ``wanted`` are read: the rest of the cluster's thousands
    take most of the time and none of the pools.
    """
    by_node = defaultdict(list)
    no_limit = []
    elsewhere = 0
    for row in rows:
        nodes = parse_node_names(row["nodelist"]) if row["nodelist"] else []
        if not nodes:
            print(f"Warning: running job {row['jobid']} has no node list, skipping")
            continue
        if wanted.isdisjoint(nodes):
            elsewhere += 1
            continue
        n_nodes = len(nodes)
        held = _row_alloc(row, max_minutes, "job")
        limit = held["minutes"]
        if row["endtime"] == "NONE" and row["timelimit"] in ("UNLIMITED", "INFINITE"):
            no_limit.append(row["jobid"])
            end = None
        elif row["endtime"] in ("N/A", "UNKNOWN", ""):
            print(f"Warning: running job {row['jobid']} has no end time, assuming its full limit")
            end = now + timedelta(minutes=limit)
        else:
            try:
                end = datetime.strptime(row["endtime"], "%Y-%m-%dT%H:%M:%S")
            except ValueError as err:
                print(f"Warning: unparsable end time for job {row['jobid']}: {err}, assuming its limit")
                end = now + timedelta(minutes=limit)
        alloc = {
            "id": row["jobid"],
            "gpus": held["gpus"] // n_nodes,
            "cpus": held["cpus"] // n_nodes,
            "mem_mb": held["mem_mb"] // n_nodes,
            "gpu_type": _gres_gpu_type(row),
            "end": end,
            "limit": limit,
            "array": held["array"],
        }
        for node in nodes:
            by_node[node].append(alloc)
    if elsewhere:
        print(f"Note: the wait replay left out {elsewhere} running jobs on nodes outside "
              "the monitored partitions")
    if no_limit:
        print(f"Note: running jobs with no time limit ({', '.join(no_limit)}) keep what they "
              f"hold for all {PLAN_HORIZON_DAYS} days of the wait replay")
    return by_node


def _competing_jobs(rows, priorities, partitions, score, gpu_type, node_gpus, max_minutes,
                    nodes, eligible_by_job):
    """Pending jobs that outrank ``score`` in any of ``partitions`` and fit this pool.

    ``rows`` are pending squeue rows, ``priorities`` comes from :func:`_sprio_by_job`,
    ``eligible_by_job`` holds the nodes each row may use anywhere. Array tasks still
    folded into their parent record are expanded so each counts once. Returns the
    list sorted by priority, highest first, and the ids of the jobs that would
    compete but whose node lists or constraint keep them out of this pool.
    """
    jobs = []
    ruled_out = set()
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
        wanted_type = _gres_gpu_type(row)
        if wanted_type and wanted_type != gpu_type:
            continue
        eligible = eligible_by_job[row["jobid"]] & nodes.keys()
        n_nodes = _first_int(row["numnodes"]) or 1
        if len(eligible) < n_nodes:
            ruled_out.add(row["jobid"])
            continue
        asked = _row_alloc(row, max_minutes, "pending job")
        gpn = -(-asked["gpus"] // n_nodes)
        if gpn > node_gpus:
            # Asks for more GPUs per node than this pool has: it will land elsewhere.
            continue
        count, throttle = 1, None
        array = asked["array"]
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
            "cpn": -(-asked["cpus"] // n_nodes),
            "mpn": -(-asked["mem_mb"] // n_nodes),
            "minutes": asked["minutes"],
            "array": array,
            "throttle": throttle,
            "eligible_nodes": eligible,
        }
        jobs.extend([job] * count)
    jobs.sort(key=lambda j: (-j["prio"], j["id"]))
    return jobs, ruled_out


def _plan_pool(nodes, running_by_node, queue, ours, now, ratio=None):
    """Plan one pool the way a backfill cycle does and return when ``ours`` starts.

    ``nodes`` maps node name to its (gpus, cpus, mem_mb) capacity, ``queue`` is
    the higher-priority pending work sorted best first, and ``ours`` is the job
    being estimated; it is planned last, as a fresh submission would be. Time is
    cut into PLAN_MINUTES buckets over PLAN_HORIZON_DAYS. Running jobs hold their
    resources until they end; then every queued job, in priority order, takes
    the earliest window in which enough of its ``eligible_nodes`` have its GPUs,
    CPUs and memory free for its whole time limit, and array tasks also respect
    their throttle.
    Lower-priority jobs fill earlier gaps only when they do not touch a window
    already claimed, which is the backfill guarantee.

    With ``ratio`` set, every job is assumed to use that fraction of its limit
    instead of all of it, which is what turns Full Limits into the Historical
    Runtime column. Returns None when the job finds no window inside the horizon.
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
            if alloc["end"] is None:
                # No time limit: its share stays taken for the whole horizon.
                end = buckets
            else:
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

    unplaceable = set()
    earliest_by_job = {}
    allowed_by_job = {}  # id(job) -> which rows of avail it may use, None for all
    for job in list(queue) + [ours]:
        # Expanded array tasks share one job dict. Once one cannot fit, later
        # copies cannot fit either: queued placements only consume resources
        # and array throttle slots within this fixed horizon.
        if id(job) in unplaceable:
            continue
        minutes = job["minutes"]
        if ratio is not None and job is not ours:
            minutes = max(minutes * ratio, 1)
        span = _bucket_span(minutes)
        if span > buckets:
            if job is ours:
                return None
            continue
        if job["throttle"]:
            use = array_use.setdefault(job["array"], np.zeros(buckets, dtype=np.int64))
        # Most jobs start early. Search the first day, then double the range
        # until a window fits or the full horizon has been checked. Each chunk
        # overlaps by span - 1 buckets so every possible start is examined.
        possible_starts = buckets - span + 1
        # Availability only falls as earlier jobs are placed, so another copy
        # of this array task cannot start before the previous copy did.
        first = earliest_by_job.get(id(job), 0)
        limit = min(first + 24 * 60 // PLAN_MINUTES, possible_starts)
        start = None
        if id(job) not in allowed_by_job:
            # None when the job may use every node of the pool, most of them do.
            eligible = job["eligible_nodes"]
            allowed_by_job[id(job)] = (
                None if len(eligible) == len(names)
                else np.array([name in eligible for name in names])[:, None]
            )
        allowed = allowed_by_job[id(job)]
        while True:
            end = limit + span - 1
            ok = (
                (avail[0, :, first:end] >= job["gpn"])
                & (avail[1, :, first:end] >= job["cpn"])
                & (avail[2, :, first:end] >= job["mpn"])
            )
            if allowed is not None:
                ok &= allowed
            if job["throttle"]:
                ok &= (use[first:end] < job["throttle"])[None, :]
            # A window fits when it holds no bad bucket.
            bad = np.cumsum(~ok, axis=1, dtype=np.int32)
            bad = np.concatenate([np.zeros((len(names), 1), dtype=bad.dtype), bad], axis=1)
            fits = (bad[:, span:] - bad[:, :-span]) == 0
            starts = np.nonzero(fits.sum(axis=0) >= job["nodes"])[0]
            if len(starts):
                relative_start = int(starts[0])
                start = first + relative_start
                candidates = np.nonzero(fits[:, relative_start])[0]
                break
            if limit == possible_starts:
                break
            first = limit
            limit = min(limit * 2, possible_starts)
        if start is None:
            if job is ours:
                return None
            # Never fits inside the horizon: it cannot delay us either.
            unplaceable.add(id(job))
            continue
        if job is ours:
            return now + timedelta(minutes=start * PLAN_MINUTES)
        earliest_by_job[id(job)] = start
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


def _account_score(profile, partition_factors):
    """Priority of a job submitted right now, per partition, for an account's
    :func:`_account_priority_profile`; None when that could not be read."""
    entry, max_qos = profile
    if entry is None or not max_qos:
        return None
    weights = _priority_weights()
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
@ttl_cache(WALLTIME_CACHE_SECONDS)
def _walltime_efficiency():
    """Measure what fraction of its requested walltime a finished GPU job really uses.

    The Full Limits replay lets every job run to its time limit; the Historical
    Runtime replay scales jobs by this ratio instead, measured per partition from
    the last few days of accounting data as elapsed time over time limit, each
    summed, so long jobs weigh more. One sacct call covers every monitored
    partition, and the result is cached because it barely moves between page
    refreshes. A failed sacct raises, so it is not cached and the next refresh
    tries again.
    """
    cmd = (
        f"sacct -a -X -S now-{WALLTIME_SAMPLE_DAYS}days -r {MONITORED_PARTITIONS} "
        "-n -P -o Partition,Elapsed,Timelimit,State,AllocTRES"
    )
    rows = parse_cmd(cmd)

    totals = defaultdict(lambda: {"elapsed": 0, "limit": 0, "jobs": 0})
    # Routine skips, tallied and reported once below.
    unfinished = cpu_only = unlimited = 0
    for row in rows:
        fields = row.split("|")
        if len(fields) < 5:
            print(f"Warning: unexpected sacct row {row!r}, skipping")
            continue
        partition, elapsed_str, limit_str, state, tres = fields[:5]
        if not state.startswith(FINISHED_STATES):
            unfinished += 1
            continue
        if "gres/gpu" not in tres:
            cpu_only += 1
            continue
        elapsed = _slurm_time_to_minutes(elapsed_str)
        limit = _slurm_time_to_minutes(limit_str)
        if not limit:
            # UNLIMITED or unparsable: there is no ratio to compute against.
            unlimited += 1
            continue
        if elapsed is None:
            print(f"Warning: sacct row {row!r} has no elapsed time, skipping")
            continue
        entry = totals[partition.strip()]
        entry["elapsed"] += elapsed
        entry["limit"] += limit
        entry["jobs"] += 1
    print(f"Note: walltime sample left out {unfinished} unfinished, {cpu_only} CPU-only and "
          f"{unlimited} unlimited jobs")

    data = {}
    for partition, entry in totals.items():
        if entry["jobs"] < WALLTIME_MIN_SAMPLE or entry["limit"] == 0:
            print(
                f"Note: only {entry['jobs']} finished GPU jobs in {partition} over the last "
                f"{WALLTIME_SAMPLE_DAYS} days, too few for a Historical Runtime estimate"
            )
            continue
        # Jobs can overrun slightly into the grace period; a ratio above 1 would
        # make Historical Runtime worse than the Full Limits replay it is scaling.
        data[partition] = {
            "ratio": min(entry["elapsed"] / entry["limit"], 1.0),
            "jobs": entry["jobs"],
        }
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


def parse_wait_estimate_to_table(job):
    """Estimate how long the requested job would wait in each GPU pool.

    One SLURM snapshot (running jobs, ranked pending jobs, node shapes) feeds a
    per-pool scheduler replay, see :func:`_plan_pool`. sbatch --test-only is
    still run once per pool, but only to learn whether the account may use it.
    """
    specs = _standard_job_specs(job)
    if not specs:
        return "<p>No GPU partitions found.</p>"

    # The SLURM reads below and the per-pool sbatch --test-only checks do not
    # depend on one another, and this panel sets how long a page load takes, so
    # overlap them. The pool is kept small: the login node is short of memory
    # to fork with, and parse_cmd already shares one run of a repeated command.
    # Work starts in the order it is submitted. The priority factors go first:
    # without the account's score there is nothing to estimate, and whatever has
    # not started by then is dropped. The two reads no other panel shares come
    # next, sacct being the slowest command here when its cache is cold, and
    # only then the test-only checks and the reads the other panels share.
    with ThreadPoolExecutor(max_workers=WAIT_ESTIMATE_WORKERS) as pool:
        factors = pool.submit(_partition_factors)
        reads = {
            "efficiency": pool.submit(_walltime_efficiency),
            "shapes": pool.submit(_node_shapes),
        }
        verdicts = [pool.submit(_test_only_check, spec, job["account"]) for spec in specs]
        reads |= {
            name: pool.submit(fetch)
            for name, fetch in (
                ("pools", _gpu_pools),
                ("partition_nodes", partition_nodes),
                ("limits", _partition_limits),
                ("priorities", _sprio_by_job),
                ("running", lambda: squeue_jobs("RUNNING")),
                ("pending", lambda: squeue_jobs("PENDING", MONITORED_PARTITIONS)),
                ("records", node_records),
            )
        }
        profile = _account_priority_profile(job["account"])
        scores = _account_score(profile, factors.result())
        if scores is None:
            pool.shutdown(cancel_futures=True)
            return (
                f"<p>Could not work out the priority a job from <b>{job['account']}</b> "
                "would get, so there is nothing to estimate against.</p>"
            )
        read = {name: future.result() for name, future in reads.items() if name != "efficiency"}
        verdicts = [future.result() for future in verdicts]
        try:
            efficiency = reads["efficiency"].result()
        except (CalledProcessError, OSError) as exc:
            # Historical Runtime only scales Full Limits, which still stands without it.
            print(f"Warning: sacct failed ({exc}); the Historical Runtime column will be blank")
            efficiency = {}
    pools = read["pools"]
    total_by_node = pools["total_by_node"]
    free_by_node = pools["free_by_node"]
    part_nodes = read["partition_nodes"]
    shapes = read["shapes"]
    limits = read["limits"]
    priorities = read["priorities"]
    now = datetime.now()
    default_limit = max(
        (l["max_minutes"] for l in limits.values() if l["max_minutes"]), default=3 * 24 * 60
    )
    eligibility = _NodeEligibility(read["records"], part_nodes)
    running_by_node = _running_jobs_by_node(read["running"], now, default_limit, eligibility.nodes)
    pending_rows = read["pending"]
    # Where each pending job may run at all, worked out once for every pool below.
    eligible_by_job = {
        row["jobid"]: eligibility.job_nodes(row, row["partition"].split(","))
        for row in pending_rows
    }
    eligibility.report("wait estimate")
    ruled_out = set()

    def _count(by_node, partition, gpu_type):
        return sum(
            entry["count"]
            for node in part_nodes.get(partition, ())
            for entry in by_node.get(node, ())
            if entry["type"] == gpu_type
        )

    rows = []
    unavailable = []
    no_access = []
    for spec, failure in zip(specs, verdicts):
        partition, gpu_type = spec["partition"], spec["gpu_type"]
        if failure is not None:
            if failure[0] == "no_access":
                no_access.append((spec, failure[1]))
            else:
                unavailable.append((spec, failure[1]))
            continue

        # The pool: reachable nodes of this partition carrying this GPU type;
        # total_by_node holds only the nodes split_by_access left open.
        nodes = {}
        for node in part_nodes.get(partition, ()):
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
        sharing = [p for p, members in part_nodes.items() if members & nodes.keys()]
        score = scores.get(partition)
        if score is None:
            print(f"Warning: no partition factor for {partition}, ranking against 0")
            score = 0
        queue, kept_out = _competing_jobs(
            pending_rows, priorities, sharing, score, gpu_type, node_gpus,
            limits.get(partition, {}).get("max_minutes") or default_limit,
            nodes, eligible_by_job,
        )
        ruled_out |= kept_out
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
            "eligible_nodes": nodes.keys(),
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

    if ruled_out:
        print(f"Note: the wait replay kept {len(ruled_out)} outranking pending jobs out of "
              "at least one pool their -w, -x or --constraint rules out")
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
                f"{spec['mem_mb'] / 1024:.1f} GiB / {walltime}"
            )
            if spec["capped"]:
                shape += (
                    f'<span class="queue-flex" title="{"; ".join(spec["capped"])}">&nbsp;*</span>'
                )
            worst_cell = _wait_cell(
                row["worst"],
                "Every job ahead of you and every running job uses its full time limit",
            )
            usage = row["usage"]
            if usage:
                typical_cell = _wait_cell(
                    row["typical"],
                    f"Every job assumed to use {usage['ratio']:.0%} of its limit: the "
                    f"{usage['jobs']} GPU jobs that finished in {partition} over the last "
                    f"{WALLTIME_SAMPLE_DAYS} days ran that share of their time limits added up",
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
                f"<tr><td><b>{gpu_type}</b> {_mem_label(gpu_type)}</td>"
                f"<td>{partition}</td>"
                f"<td class='queue-muted'>{shape}</td>"
                f"<td>{get_resource_bar(row['free'], row['capacity'])}</td>"
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
        "Full Limits</th>"
        '<th title="Same replay with every job cut to the fraction of its limit jobs in '
        'this partition actually use">Historical Runtime</th>'
        '<th title="Model start from Historical Runtime, or Full Limits when no sample '
        'is available. This is not a Slurm forecast.">Model Start</th></tr>'
    )

    gpu_rows, mig_rows = _split_mig(rows, lambda row: row["spec"]["gpu_type"])
    owner_entry = profile[0]
    if owner_entry["member"]:
        whose = (f"a new job by <b>{owner_entry['user']}</b> under <b>{job['account']}</b>, "
                 "using your own fair-share")
    else:
        whose = (f"a new job under <b>{job['account']}</b>, using the median fair-share of "
                 f"its members, since <b>{owner_entry['user']}</b> has no association there,")

    table_html = (
        f"<table>{header}{_render_rows(gpu_rows)}</table>"
        f'<p class="queue-note">These are per-pool model estimates for {whose} '
        "and zero waiting age. They are not Slurm start forecasts or guaranteed bounds. "
        "<b>Full Limits</b> uses requested walltimes; <b>Historical Runtime</b> scales "
        "other jobs by the share of their time limits, added up, that the partition's GPU "
        f"jobs used over the last {WALLTIME_SAMPLE_DAYS} days, failed and cancelled ones "
        "included, so long jobs weigh more than short ones. Each row assumes all eligible higher-priority "
        "work uses this pool; flexible jobs may actually start elsewhere, so a small pool "
        "can show a much longer wait than the job experiences. "
        "Future arrivals, reservations, changing priorities and user or array limits "
        "can change the result. Jobs currently blocked by dependencies or user limits "
        "are omitted. The Waiting Queue shows Slurm's own estimates for submitted jobs. "
        "Free Now counts unallocated GPUs; CPU, memory and the requested duration must "
        "also fit. A <b>*</b> means the partition limits reduced the requested shape.</p>"
    )
    if mig_rows:
        table_html += _folded("MIG slices", header + _render_rows(mig_rows))
    if unavailable:
        # These failed on the job shape the form asked for, so the reason is
        # actionable: shrink the request, or accept that the pool cannot host it.
        rows_html = "".join(
            f"<tr><td><b>{spec['gpu_type']}</b></td><td>{spec['partition']}</td>"
            f"<td class='queue-muted'>{error}</td></tr>"
            for spec, error in unavailable
        )
        table_html += _folded(
            f"Pools this job shape does not fit ({len(unavailable)})",
            _table_row(["GPU Type", "Partition", "Reason"], "th") + rows_html,
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


def _submission_qos(default, allowed, who, qos_priorities):
    """The QoS a job of association ``who`` gets without --qos, or None (warned)
    when that cannot be told."""
    names = []
    for name in (q for q in allowed.split(",") if q):
        if name in qos_priorities:
            names.append(name)
        else:
            print(f"Warning: QOS {name!r} of {who} has no configured priority, skipping")
    if default:
        if default in names:
            return default
        print(f"Warning: default QOS {default!r} of {who} is not among its usable "
              f"QOS ({', '.join(names) or 'none'})")
        return None
    if len(names) == 1:
        return names[0]
    if "normal" in names:
        # Several QoS and no default. From memory of slurmctld, not checked here
        # (every association seen on 2026-10-19 allowed one): a job without --qos
        # then gets the root association's default, unset on Rivanna, or normal.
        print(f"Warning: {who} allows {', '.join(names)} with no default QOS, assuming normal")
        return "normal"
    return None


def _account_priority_profile(account):
    """The monitor owner's fair-share and submission QoS under ``account``.

    Returns ``(entry, max_qos)``. ``entry`` holds the owner's FairShare and the QoS a
    job of theirs gets without --qos, with that QoS's priority, or is None (with a
    warning) when SLURM gives no FairShare or no usable QoS; ``max_qos`` is the
    highest QoS priority configured on the cluster. When the owner has no
    association under ``account`` (``entry["member"]`` False), the median FairShare
    of its members and the account's own QoS stand in, so the panels still show
    where a typical member's job would land.
    """
    owner = _app_owner()
    shares = {}
    for row in parse_cmd(f"sshare -a -n -P -A {account} -o Account,User,FairShare"):
        fields = row.split("|")
        if len(fields) < 3:
            print(f"Warning: unexpected sshare row {row!r}, skipping")
            continue
        row_account, user, share = (f.strip() for f in fields[:3])
        # The account-level row carries no FairShare of its own; only users do.
        if not user:
            continue
        if row_account != account:
            print(f"Warning: sshare -A {account} returned a row for {row_account}/{user}, skipping")
            continue
        if not re.fullmatch(r"\d+(?:\.\d+)?", share):
            print(f"Warning: no FairShare for {row_account}/{user}, skipping")
            continue
        shares[user] = float(share)

    member = owner in shares
    fairshare = None
    if member:
        fairshare = shares[owner]
    elif shares:
        ranked = sorted(shares.values())
        fairshare = ranked[len(ranked) // 2]
        print(f"Warning: {owner} has no FairShare under {account}, using the median of its "
              f"{len(ranked)} members")

    qos_priorities = _qos_priorities()
    assocs = {}  # user ("" for the account itself) -> (DefaultQOS, allowed QoS)
    for row in parse_cmd(
        f"sacctmgr -n -P show assoc account={account} format=Account,User,DefaultQOS,QOS"
    ):
        fields = row.split("|")
        if len(fields) < 4:
            print(f"Warning: unexpected sacctmgr assoc row {row!r}, skipping")
            continue
        row_account, user, default, allowed = (f.strip() for f in fields[:4])
        if row_account != account:
            print(f"Warning: sacctmgr account={account} returned a row for "
                  f"{row_account}/{user}, skipping")
            continue
        assocs[user] = (default, allowed)
    # A non-member gets the account's own settings, which its members inherit
    # unless their association overrides them.
    key, who = (owner, f"{account}/{owner}") if member else ("", account)
    if key in assocs:
        qos = _submission_qos(*assocs[key], who, qos_priorities)
    else:
        print(f"Warning: sacctmgr has no association for {who}")
        qos = None

    max_qos = max(qos_priorities.values()) if qos_priorities else 0
    if fairshare is None:
        print(f"Warning: no user of {account} has a FairShare value")
        return None, max_qos
    if qos is None:
        print(f"Warning: no unambiguous submission QOS for {who}")
        return None, max_qos
    entry = {"user": owner, "member": member, "fairshare": fairshare, "qos": qos,
             "qos_priority": qos_priorities[qos]}
    return entry, max_qos


def _partition_priority_data():
    """Per partition: the priorities of the jobs competing there, and its factor.

    Only GPU jobs of the monitored partitions waiting for resources or priority
    count, each array task once: jobs held back by a dependency or a limit are
    not racing us for a node right now.
    """
    totals = defaultdict(list)
    factors = {}
    pending = {row["jobid"]: row for row in squeue_jobs("PENDING", MONITORED_PARTITIONS)
               if _classify_pending_reason(row["reason"]) == "waiting"
               and alloc_gpu_count(row["tres-alloc"])}
    # Routine skips, tallied and reported once below.
    not_competing = 0
    for jobid, partition, total, factor in _sprio_rows():
        if not total.isdigit() or not factor.isdigit():
            print(f"Warning: unparsable sprio numbers {total!r}/{factor!r} for job {jobid}, skipping")
            continue
        if jobid not in pending:
            not_competing += 1
            continue
        totals[partition].extend([int(total)] * _array_task_count(pending[jobid]["arraytaskid"]))
        if partition in factors and factors[partition] != int(factor):
            print(
                f"Warning: partition {partition} reports two priority factors "
                f"({factors[partition]} and {factor}); keeping the first"
            )
        else:
            factors[partition] = int(factor)
    if not_competing:
        print(f"Note: priority ranks left out {not_competing} sprio rows of jobs that are held "
              "back, ask for no GPU or are queued outside the monitored partitions")
    for partition in totals:
        totals[partition].sort()
    return totals, factors


def parse_priority_to_table(account):
    """Rank a freshly submitted job from ``account`` inside every GPU pool."""
    profile = _account_priority_profile(account)
    entry, max_qos = profile
    if entry is None or not max_qos:
        return f"<p>Could not read the priority settings of <b>{account}</b> from SLURM.</p>"
    totals, factors = _partition_priority_data()

    partitions = [p for p in partition_nodes() if p in totals]
    partitions.sort(key=lambda p: factors.get(p, 0), reverse=True)
    if not partitions:
        return "<p>No GPU partition currently has jobs competing for priority.</p>"
    # A job submitted right now has no age factor yet.
    scores = _account_score(profile, {p: factors.get(p, 0) for p in partitions})
    if entry["member"]:
        whose = f"{entry['user']} / {account}"
        fairshare_text = (
            f"Fair-share is specific to <b>{entry['user']}</b> under <b>{account}</b>; "
            "members of the same account can have different priorities."
        )
    else:
        whose = f"{account} median"
        fairshare_text = (
            f"<b>{entry['user']}</b> has no association under <b>{account}</b>, so this "
            "uses the median fair-share of its members, who can each have a different one."
        )

    header = (
        '<tr><th>Partition</th><th title="Pending jobs SLURM is currently ranking in '
        'this partition; jobs blocked by a dependency get no priority and are not '
        f'counted">Competing Tasks</th><th>{whose}<br><span class="queue-muted">'
        f'{entry["qos"]} / fs {entry["fairshare"]:.6f}</span></th></tr>'
    )

    body = ""
    for partition in partitions:
        values = totals[partition]
        score = scores[partition]
        rank = bisect.bisect_left(values, score) / len(values) * 100
        if rank >= 75:
            css = "queue-pressure-low"
        elif rank >= 40:
            css = "queue-pressure-mid"
        else:
            css = "queue-pressure-high"
        cells = (
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
        f"{fairshare_text} The new job has zero waiting age and Nice=0. Existing jobs retain their age bonus, which can "
        "decide the order even within one lab. Counts include every pending array task "
        "currently waiting for resources or priority; dependency and limit blocks are "
        "excluded. This percentile compares priority scores, not resource fit or a "
        "guaranteed place in Slurm's schedule.</p>"
    )
    return f"<table>{header}{body}</table>{note}"

def _short_stamp(stamp):
    """2026-09-17T16:15:17 -> 09-17 16:15; N/A and the like pass through."""
    match = re.fullmatch(r"\d{4}-(\d\d-\d\d)T(\d\d:\d\d):\d\d", stamp)
    return f"{match.group(1)} {match.group(2)}" if match else stamp


def _short_limit(text):
    """3-00:00:00 -> 3d, 1-12:00:00 -> 1d12h, 01:30:00 -> 1h30m.

    Keep minutes: a 90-minute allocation must not be shown as two hours.
    """
    minutes = _slurm_time_to_minutes(text)
    if minutes is None:
        return text
    days, remainder = divmod(minutes, 24 * 60)
    parts = ((days, "d"), *zip(divmod(remainder, 60), "hm"))
    return "".join(f"{value}{unit}" for value, unit in parts if value) or "0m"


def _display_job_id(job):
    """Use Slurm's array parent and task index rather than its internal row ID."""
    tasks = job["arraytaskid"]
    if tasks in UNSET_JOB_FIELD:
        return job["jobid"]
    suffix = _short_tasks(tasks)
    if not tasks.isdigit():
        suffix = f"[{suffix}]"
    return f"{job['arrayjobid']}_{suffix}"


def parse_queue_to_table(account):
    """Pending jobs of ``account``, laid out like squeue's own columns.

    START_TIME is SLURM's own estimate (what ``squeue --start`` reports), filled
    in by the backfill scheduler; it reads N/A for jobs it has not planned, such
    as held ones. State and elapsed time are left out because every row here is
    PENDING at 0:00. GPUS is the total requested by one job or array task, read
    from its allocated TRES. Both timestamps drop the year to keep the table narrow.

    squeue folds the not yet started tasks of an array into one row; its JOBID
    carries their range the way squeue prints it, e.g. ``20478034_[155-742%4]``,
    with a long list cut short (see :func:`_short_tasks`). The last line totals the account's pending jobs and the GPUs they ask for,
    counting each task of such a row.
    """
    columns = ("arraytaskid", "partition", "username", "timelimit",
               "tres-alloc", "submittime", "starttime", "reason")
    jobs = squeue_jobs("PENDING", account=account)
    if not jobs:
        return f"no pending job in {account}"

    shown = []
    total_jobs = total_gpus = 0
    for job in jobs:
        tasks, partition, user, limit, tres, submit, start, reason = [job[c] for c in columns]
        count = _array_task_count(tasks)
        jobid = _display_job_id(job)
        gpus = alloc_gpu_count(tres)
        total_jobs += count
        total_gpus += count * gpus
        shown.append((jobid, partition, user, _short_limit(limit), gpus, submit, start, reason))

    id_width = max(10, *(len(job[0]) for job in shown))
    limit_width = max(len("LIMIT"), *(len(job[3]) for job in shown))
    line = ("{:>" + str(id_width) + "} {:>14} {:>8} {:>" + str(limit_width)
            + "} {:>5} {:>11} {:>11}  {}")
    out = [line.format("JOBID", "PARTITION", "USER", "LIMIT", "GPUS",
                       "SUBMIT_TIME", "START_TIME", "NODELIST(REASON)")]
    for jobid, partition, user, limit, gpus, submit, start, reason in shown:
        # A job queued on several partitions lists all of them; cut it to the
        # column like squeue itself would rather than stretching every row.
        out.append(line.format(jobid, partition[:14], user, limit, gpus,
                               _short_stamp(submit), _short_stamp(start), reason))
    out.append(f"\n{total_jobs} pending jobs, {total_gpus} GPUs")
    return "\n".join(out)


def _gpu_type_on_nodes(jobid, node_str, inventory):
    """GPU type a running job holds, read off the nodes it runs on.

    A job sent to a mixed partition such as ``gpu`` without naming a type has no
    type anywhere in squeue; only the node it landed on tells. ``inventory`` is
    :func:`parse_all_gpus` output. Returns ``?`` when the node list is missing or
    a node is missing from the inventory, and every type joined with ``/`` when
    the nodes offer more than one, since the job's own share cannot be told
    apart then.
    """
    nodes = parse_node_names(node_str) if node_str else []
    if not nodes:
        print(f"Warning: GPU job {jobid} has no node list; type shown as ?")
        return "?"
    types = set()
    for node in nodes:
        if node not in inventory:
            print(f"Warning: GPU job {jobid} runs on {node}, which lists no GPUs; type shown as ?")
            return "?"
        types.update(entry["type"] for entry in inventory[node])
    if len(types) > 1:
        print(f"Warning: nodes {node_str} of job {jobid} offer several GPU types "
              f"{sorted(types)}; showing them all")
    return "/".join(sorted(types, key=_type_rank))


def parse_running_to_table(account):
    """Running jobs of ``account``, the counterpart of the waiting queue.

    TIME is how long the job has run and START_TIME when it started, both as
    squeue reports them. GPUS is what the job holds, read from its allocated
    TRES, and GPU_TYPE the type of the nodes it runs on (see
    :func:`_gpu_type_on_nodes`). The last line totals the account's jobs and GPUs.
    """
    columns = ("partition", "username", "timeused", "timelimit",
               "tres-alloc", "starttime", "nodelist")
    jobs = squeue_jobs("RUNNING", account=account)
    if not jobs:
        return f"no running job in {account}"

    inventory = parse_all_gpus()
    shown = []
    for job in jobs:
        partition, user, used, limit, tres, start, nodes = [job[c] for c in columns]
        jobid = _display_job_id(job)
        gpus = alloc_gpu_count(tres)
        gpu_type = _gpu_type_on_nodes(jobid, nodes, inventory) if gpus else "-"
        shown.append((jobid, partition, user, used, _short_limit(limit), gpus, gpu_type,
                      start, nodes))

    # A running job sits in one partition, so the column is as wide as the
    # longest name rather than cut short like the waiting queue's.
    part_width = max(len("PARTITION"), *(len(job[1]) for job in shown))
    type_width = max(len("GPU_TYPE"), *(len(job[6]) for job in shown))
    id_width = max(len("JOBID"), *(len(job[0]) for job in shown))
    limit_width = max(len("LIMIT"), *(len(job[4]) for job in shown))
    line = ("{:>" + str(id_width) + "} {:>" + str(part_width) + "} {:>8} {:>11} {:>"
            + str(limit_width) + "} {:>5} {:>" + str(type_width) + "} {:>11}  {}")
    out = [line.format("JOBID", "PARTITION", "USER", "TIME", "LIMIT", "GPUS",
                       "GPU_TYPE", "START_TIME", "NODELIST")]
    total_gpus = 0
    for jobid, partition, user, used, limit, gpus, gpu_type, start, nodes in shown:
        total_gpus += gpus
        out.append(line.format(jobid, partition, user, used, limit,
                               gpus, gpu_type, _short_stamp(start), nodes))
    out.append(f"\n{len(jobs)} running jobs, {total_gpus} GPUs")
    return "\n".join(out)


def _table_row(cells, tag="td"):
    """One HTML table row with a ``tag`` cell per entry of ``cells``."""
    return "<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>"


def _folded(summary, rows):
    """A table folded away under a clickable ``summary`` line."""
    return f'<details class="mig-details"><summary>{summary}</summary><table>{rows}</table></details>'


@ttl_cache(CMD_CACHE_SECONDS)
def _hdquota():
    """hdquota's output, reused for CMD_CACHE_SECONDS like a SLURM query's."""
    return check_output("hdquota", shell=True).decode("utf-8")


def parse_disk_quota():
    """Run 'hdquota' command and parse the output to an HTML table."""
    try:
        output = _hdquota()
    except (CalledProcessError, OSError) as exc:
        print(f"Warning: hdquota failed: {exc}")
        return f"<p>Error getting disk quota information: {exc}</p>"

    headers = ["Storage Type", "Location", "Size", "Used", "Avail", "Use%"]
    table_html = "<table>" + _table_row(headers, "th")
    for line in output.split("\n")[2:]:  # Skip header and separator lines
        parts = line.split()
        if not parts:
            continue
        # The storage type is one word ("Scratch") or two ("Home Directory").
        if len(parts) > 1 and parts[1] in ("Directory", "Project", "Standard"):
            parts = [f"{parts[0]} {parts[1]}"] + parts[2:]
        if len(parts) < 9:
            print(f"Warning: unexpected hdquota line {line!r}, skipping")
            continue
        size, used, avail = (f"{parts[i]} {parts[i + 1]}" for i in (2, 4, 6))
        table_html += _table_row((parts[0], parts[1], size, used, avail, parts[8]))
    return table_html + "</table>"


def parse_allocations_to_table(account):
    """Run 'allocations -a <account>' and parse the output to an HTML table.

    The command only prints the balance table to members of the account; for
    everyone else it lists the members and nothing else, which is why an empty
    table is reported as "not a member" rather than "no allocation".
    """
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

    columns = ["Description", "StartTime", "EndTime", "Allocated", "Remaining",
               "PercentUsed", "Active"]
    table_html = "<table>" + _table_row(columns, "th")
    # Split on whitespace. StartTime is a date and a time, so a complete row has
    # eight fields for the seven columns.
    for row in allocation_lines:
        parts = row.split()
        if len(parts) < 8:
            print(f"Warning: incomplete allocation row {row!r} for {account}, skipping")
            continue
        table_html += _table_row((parts[0], f"{parts[1]} {parts[2]}", *parts[3:8]))
    return table_html + "</table>"


# ---------------------------------------------------------------------------
# "My running jobs": live CPU, memory and GPU readings for the jobs belonging
# to whichever account started this app.  Only its own jobs can be read, since
# both sstat and a --jobid step are refused for someone else's allocation.
# ---------------------------------------------------------------------------

# The job name is free text and so goes last, see squeue_fields.
MY_JOBS_FORMAT = ("jobid", "partition", "timeused", "timelimit", "tres-alloc", "name")
# Longest job name shown under its id; longer ones are cut to keep the column narrow.
JOB_NAME_WIDTH = 48
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


def _my_running_jobs():
    """One dict per running job of this account, in squeue's own order."""
    return squeue_fields(f"-a -h -u {_app_owner()} -t RUNNING", MY_JOBS_FORMAT)


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

        cpu = _slurm_time_to_seconds(tres_field(ave, "cpu") or "")
        if cpu is not None:
            entry["cpu_seconds"] += cpu
        for key, tres in (("mem_mb", ave), ("peak_mem_mb", peak)):
            mem = tres_field(tres, "mem")
            if mem is None:
                continue
            mem_mb = mem_to_mb(mem)
            if mem_mb is None:
                print(f"Warning: unparsable memory {mem!r} for step {step}, ignoring it")
                continue
            entry[key] += mem_mb

        util = tres_field(ave, "gres/gpuutil")
        if util is not None:
            if util.isdigit():
                entry["gpu_util"] = max(entry["gpu_util"] or 0, int(util))
            else:
                print(f"Warning: unparsable gres/gpuutil {util!r} for step {step}, ignoring it")
        gpu_mem = tres_field(ave, "gres/gpumem")
        if gpu_mem is not None:
            gpu_mem_mb = mem_to_mb(gpu_mem)
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


# What _gpu_cells reads off one GPU. A source that lacks a reading leaves it None;
# the windows and sample counts say what an average was taken over.
GPU_ROW_KEYS = ("index", "name", "util", "util_window_s", "util_n", "mem_used_mb",
                "mem_total_mb", "power", "power_window_s", "power_n", "power_limit", "temp")


def _gpu_row(source, **readings):
    """One GPU's readings as _gpu_cells takes them, the rest None."""
    unknown = set(readings) - set(GPU_ROW_KEYS)
    if unknown:
        raise TypeError(f"unknown GPU readings {sorted(unknown)}")
    row = dict.fromkeys(GPU_ROW_KEYS)
    row.update(readings, source=source)
    if row["name"] is not None:
        row["name"] = re.sub(r"^NVIDIA\s+", "", row["name"])
    return row


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
        gpus.append(_gpu_row(
            "nvml",
            index=card["index"],
            name=card["name"] or "GPU",
            util=card["util_avg"],
            util_window_s=card["util_window_s"],
            util_n=card["util_n"],
            mem_used_mb=card["mem_used_mb"],
            mem_total_mb=card["mem_total_mb"],
            power=card["power_avg_w"],
            power_window_s=card["power_window_s"],
            power_n=card["power_n"],
            power_limit=card["power_limit_w"],
            temp=card["temp_c"],
        ))
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
        gpus.append(_gpu_row(
            "nvidia-smi",
            index=index,
            name=name,
            util=_smi_number(util, jobid, "utilization.gpu"),
            mem_used_mb=_smi_number(mem_used, jobid, "memory.used"),
            mem_total_mb=_smi_number(mem_total, jobid, "memory.total"),
            power=_smi_number(power, jobid, "power.draw"),
            power_limit=_smi_number(power_limit, jobid, "power.limit"),
            temp=_smi_number(temp, jobid, "temperature.gpu"),
        ))
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
    return [_gpu_row(
        "accounting",
        util=counters["gpu_util"],
        util_window_s=ACCOUNTING_SAMPLE_SECONDS,
        mem_used_mb=counters["gpu_mem_mb"],
    )]


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
    sampled; then one short gpu_probe.py step per GPU job (nvidia-smi where that
    fails) gives the readings SLURM does not keep, above all the power draw and
    the board temperature.  A job whose probe fails still shows the GPU
    utilisation and memory that sstat collected, marked as coming from
    accounting rather than from the card.
    """
    owner = _app_owner()
    jobs = _my_running_jobs()
    if not jobs:
        return (
            f'<p class="queue-note">{escape(display_name(owner))} has no running '
            "job right now.</p>"
        )

    jobids = [job["jobid"] for job in jobs]
    gpu_jobs = [job["jobid"] for job in jobs if alloc_gpu_count(job["tres-alloc"])]
    # The two wait on different things -- sstat on the accounting plugin, the
    # probes on the nodes running the jobs -- and neither needs the other, so
    # overlap them instead of paying for both in turn.  One extra thread next to
    # the pool the probes already use.
    with ThreadPoolExecutor(max_workers=1) as reader:
        counters = reader.submit(_sstat_usage, jobids)
        probes = _probe_gpus(gpu_jobs)
        # result() re-raises, so an sstat failure still reaches slurm_response.
        usage = counters.result()

    # Jobs that have ended since the last refresh must not keep a sample around,
    # or a recycled job id would be handed someone else's counter.
    for stale in set(_CPU_SAMPLES) - set(jobids):
        del _CPU_SAMPLES[stale]

    now = time.time()
    columns = ("JOB", "PARTITION", "ELAPSED", "CPU", "RAM",
               "GPU", "UTIL", "GPU MEM", "POWER", "TEMP")
    html = ["<table>", _table_row(columns, "th")]

    stale_cpu = False
    accounting_only = False
    instant_only = False
    for job in jobs:
        jobid = job["jobid"]
        tres = job["tres-alloc"]
        counters = usage.get(jobid)
        alloc_cpus = _tres_count(tres_field(tres, "cpu"))
        alloc_mem_mb = mem_to_mb(tres_field(tres, "mem") or "")
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
                    f'{escape(job["name"][:JOB_NAME_WIDTH].rstrip())}</span></td>'
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
    # Deployed with stdout sent to app.log, where Python holds prints back until
    # 8 KB pile up, and every restart or job end kills the process before then:
    # the log had not one of the app's warnings or notes. Write each line out.
    sys.stdout.reconfigure(line_buffering=True)

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(
            open("index.html").read(),
            # The node this is serving from, not the bind address: it is what
            # goes in the OnDemand URL (/rnode/<host>/<port>/) that reaches it,
            # and "0.0.0.0" told nobody anything.
            hostname=f"{socket.gethostname()}:{args.port}",
            accounts=LAB_ACCOUNTS,
            my_jobs_min_gap_ms=CPU_SAMPLE_MIN_GAP * 1000,
            standard_hours=STANDARD_JOB_HOURS,
        )

    @app.route("/time_feed")
    def time_feed():
        now = datetime.now(pytz.timezone("America/New_York"))
        return Response(f'updated at: {now.strftime("%Y.%m.%d | %H:%M:%S")}', mimetype="text")

    # The panels that are one builder each: (endpoint, builder, follows the
    # account picker). index.html fetches them by these endpoint names.
    for endpoint, build, by_account in (
        ("resource", parse_usage_to_table, False),
        ("queue", parse_queue_to_table, True),
        ("running", parse_running_to_table, True),
        ("queue_stats", parse_queue_stats_to_table, False),
        ("priority", parse_priority_to_table, True),
        ("leaderboard", parse_leaderboard, False),
        ("disk_quota", parse_disk_quota, False),
        ("allocations", parse_allocations_to_table, True),
        ("my_jobs", parse_my_jobs_to_table, False),
    ):
        def panel(build=build, by_account=by_account):
            account = (selected_account(request.args),) if by_account else ()
            return slurm_response(build, *account)

        app.add_url_rule(f"/{endpoint}", endpoint, panel)

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

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
