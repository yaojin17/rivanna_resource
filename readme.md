# Slurm Website Monitor for UVA Rivanna

A web dashboard for the [SLURM](https://slurm.schedmd.com/documentation.html) scheduler on
UVA's Rivanna cluster. On a single page it shows what the cluster is doing, where your
account stands in the queue, and what your own running jobs are consuming.

The project extends the monitor originally written for the
[Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/), Oxford, and is maintained by
the UVA Computer Vision Lab.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
  - [Access through Open OnDemand](#access-through-open-ondemand)
  - [Access through an SSH tunnel](#access-through-an-ssh-tunnel)
  - [Keeping the monitor running](#keeping-the-monitor-running)
  - [Troubleshooting](#troubleshooting)
- [Configuration](#configuration)
- [Dashboard Reference](#dashboard-reference)
- [Command-Line Tool](#command-line-tool)
- [Changelog](#changelog)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

The dashboard is organised around three questions:

| Question | Panels |
| --- | --- |
| What is the cluster doing? | Resource available, GPU Leaderboard, Queue Overview |
| Where does my account stand? | Allocations, Estimated Wait, Priority in Each Pool, Waiting Queue |
| What are my own jobs doing? | My Running Jobs, Disk Quota |

Design principles:

- **On-demand refresh.** Panels load when the page opens, when its tab regains focus, and
  on *Update All*. Nothing polls in the background.
- **Independent panels.** All panels are requested concurrently, so a slow or failed panel
  does not affect the others. A full refresh takes a few seconds.
- **Low scheduler load.** Identical SLURM queries issued within ten seconds are answered
  once and shared between panels.
- **No tunnel required.** The page can be served through Open OnDemand behind NetBadge
  authentication.

## Requirements

- **A Rivanna login node.** Every panel is built from SLURM client commands (`squeue`,
  `sinfo`, `scontrol`, `sacct`, `sprio`, `sshare`) and the site tools `allocations` and
  `hdquota`.
- **Python 3** (tested with 3.11) and the packages listed in `requirements.txt`.
- **Your own account.** The monitor runs as the user who starts it. Panels about "you"
  (*My Running Jobs*, *Disk Quota*, the highlighted leaderboard row) describe that user;
  SLURM does not report another user's job steps.

## Quick Start

```sh
git clone https://github.com/UVA-Computer-Vision-Lab/rivanna_resource.git
cd rivanna_resource
pip install -r requirements.txt
python app.py
```

The monitor listens on `0.0.0.0:2070` by default; `--host` and `--port` override this.
The page title shows the node and port it is serving from. See [Deployment](#deployment)
for how to reach it from your own machine.

## Deployment

Login nodes have internal addresses only, so the page cannot be opened directly from
outside the cluster. Two access paths are supported.

### Access through Open OnDemand

Rivanna's Open OnDemand instance proxies a port on a cluster node at:

```
https://ood.hpc.virginia.edu/rnode/<node>/<port>/
```

`<node>` is the host running the monitor (the output of `hostname`) and `<port>` is the
port it listens on. Access is protected by NetBadge and requires no setup on the client.

> [!NOTE]
> The path is `rnode`. Rivanna does not expose the `/node/` variant, which returns
> *Not Found*.

### Access through an SSH tunnel

```sh
ssh -L 2070:<node>:2070 <computing-id>@login.hpc.virginia.edu
```

Then open `http://localhost:2070`. Name the node explicitly: `login.hpc.virginia.edu`
resolves to several login nodes, so a session may land on a different node from the one
running the monitor. The monitor must listen on `0.0.0.0` (the default) for this to work.

### Keeping the monitor running

A monitor started from a terminal stops when that terminal closes. Rivanna does not permit
users to run `crontab`, `at`, or lingering `systemd --user` services, so persistence is
handled by a hook in the shell startup file.

**1. Choose a port.** Only one process can listen on a given port per node, so users who
share a login node need different ports. Any free port above 1024 will do.

**2. Create `~/.config/rivmon.sh`**, setting `RIVMON_DIR` to your clone and `RIVMON_PORT`
to your port:

```sh
RIVMON_DIR="$HOME/rivanna_resource"   # where you cloned this repository
RIVMON_PORT=2070                      # a port of your own; see above

# The monitor's URL on this node, with a warning if you are not serving it.
rivmon() {
    printf 'https://ood.hpc.virginia.edu/rnode/%s/%s/\n' "$(hostname)" "$RIVMON_PORT"
    ss -lntpH "sport = :$RIVMON_PORT" 2>/dev/null | grep -q 'users:((' || {
        echo "rivmon: you are not serving port $RIVMON_PORT on $(hostname)" >&2
        return 1
    }
}

if [[ $- == *i* ]] && [ -d "$RIVMON_DIR" ]; then
    if ss -lntpH "sport = :$RIVMON_PORT" 2>/dev/null | grep -q 'users:(('; then
        echo "Rivanna monitor: $(rivmon)"
    elif ss -lntH "sport = :$RIVMON_PORT" 2>/dev/null | grep -q .; then
        echo "Rivanna monitor: port $RIVMON_PORT is taken by someone else on" \
             "$(hostname); set RIVMON_PORT to another" >&2
    else
        ( cd "$RIVMON_DIR" && setsid nohup python app.py --host 0.0.0.0 \
            --port "$RIVMON_PORT" >> "$RIVMON_DIR/app.log" 2>&1 < /dev/null & ) \
            >/dev/null 2>&1
        echo "Rivanna monitor: https://ood.hpc.virginia.edu/rnode/$(hostname)/$RIVMON_PORT/"
    fi
fi
```

**3. Source it** from `~/.bashrc`, and from `.zshrc` as well if you use zsh:

```sh
[ -f "$HOME/.config/rivmon.sh" ] && . "$HOME/.config/rivmon.sh"
```

Every interactive shell then starts the monitor, unless you are already running one on
that node, and prints its OnDemand URL. The `rivmon` command prints the URL again at any
time.

How the hook behaves:

- **Survives logout.** `setsid` places the server in a session of its own.
- **Recovers after a reboot.** The next interactive shell on that node starts it again.
- **One instance per node.** Login nodes are assigned round-robin. Each node you use runs
  its own copy, and the URL differs from node to node.
- **Checks ownership.** Only a listener you own counts as running. If another user holds
  the port, the hook reports it instead of printing their URL as yours.
- **Leaves non-interactive shells alone.** Nothing runs outside an interactive shell, so
  `scp`, `rsync` and `ssh <host> <command>` behave as before.
- **Logs to a file.** Output is appended to `app.log` in the repository, which git ignores.
- **Uses your login environment.** The `python` on your `PATH` at login must provide the
  packages in `requirements.txt`.

### Troubleshooting

| Symptom | Cause | Resolution |
| --- | --- | --- |
| OnDemand returns *Not Found* for the whole page | The URL uses `/node/` | Use `/rnode/<node>/<port>/` |
| OnDemand shows an error instead of the dashboard | Nothing is listening on that node and port | Run `rivmon` on that node; it reports whether you are serving the port |
| The page loads but every panel shows *Not Found* | A checkout older than 2026-09-18 requests panels from the site root, which the proxy does not rewrite | Update the repository; panels are now requested relative to the page |
| The hook reports that the port is taken | Another user is listening on that port on the same login node | Set `RIVMON_PORT` to a different port |
| A saved URL stops working | A new session landed on a different login node | Run `rivmon` for the current URL, or keep using the earlier node's URL while that instance is up |
| The monitor is gone after maintenance | The login node was rebooted | Open an interactive shell on that node; the hook starts it again |

## Configuration

Settings are environment variables read when `app.py` starts.

| Variable | Default | Description |
| --- | --- | --- |
| `RIVANNA_ACCOUNTS` | `uva_cv_lab,cang-lab-in-silico` | Accounts offered by the header picker, comma separated. The first is the default. |
| `RIVANNA_STANDARD_HOURS` | `24` | Walltime, in hours, that the wait-estimate form starts with. |
| `RIVANNA_WALLTIME_DAYS` | `3` | Days of finished jobs sampled to measure what fraction of its time limit a job really uses (the `Typical` column). |
| `RIVANNA_WALLTIME_TTL` | `1800` | Seconds that sample is reused before `sacct` is queried again. |
| `RIVANNA_TEST_ONLY_TTL` | `60` | Seconds an account's `sbatch --test-only` verdict for a pool is kept. |
| `RIVANNA_CMD_CACHE_SECONDS` | `10` | Seconds an identical SLURM query is shared between panels; `0` disables sharing. `sstat` is never shared, because the live CPU figure subtracts two readings. |

The header, footer and styling live in [index.html](index.html) and can be adapted to
your group.

## Dashboard Reference

The header holds an **account picker**, *Update All*, *Unfold All* and a dark mode toggle.
Panels scoped to an account follow the picker, so the page shows one account at a time;
the choice is remembered across visits. Sections start collapsed, and the page remembers
which ones you opened.

| Panel | Scope | Source |
| --- | --- | --- |
| [My Running Jobs (live)](#my-running-jobs-live) | You | `squeue`, `sstat`, `srun` |
| [Resource available](#resource-available) | Cluster | `sinfo`, `squeue`, `scontrol` |
| [GPU Leaderboard](#gpu-leaderboard) | Cluster | `sinfo`, `squeue` |
| [Disk Quota](#disk-quota) | You | `hdquota` |
| [Allocations](#allocations) | Account | `allocations` |
| [Queue Overview (All Users)](#queue-overview-all-users) | Cluster | `sinfo`, `squeue`, `scontrol` |
| [Estimated Wait for a Standard Job](#estimated-wait-for-a-standard-job) | Account | `squeue`, `sprio`, `sshare`, `sacct`, `sacctmgr`, `scontrol`, `sinfo`, `sbatch --test-only` |
| [Priority in Each Pool](#priority-in-each-pool) | Account | `sprio`, `sshare`, `sacctmgr`, `scontrol`, `sinfo` |
| [Waiting Queue](#waiting-queue) | Account | `squeue` |

*Scope:* **Cluster** covers all users, **Account** is the account chosen in the picker,
and **You** is the user running `app.py`. Panels are listed in page order.

### My Running Jobs (live)

Resource consumption of your running jobs: busy cores against reserved cores, resident
memory against reserved memory, and per GPU its utilisation, memory, power draw and
temperature.

- CPU and memory come from a single `sstat` call covering all jobs.
- GPU figures are read on the card by a short step inside each job (`srun --overlap`),
  which also limits the reading to the GPUs that job holds.
- GPU utilisation and power are averages, not snapshots. One `utilization.gpu` reading
  covers 1/6--1 s and swings between 45% and 100% on a steady job, so `gpu_probe.py` reads
  the samples the driver has already buffered (about 5 s of utilisation and 2.4 s of
  power) at no extra wait. Memory and temperature are point readings. Hover a bar to see
  its window and sample count.
- Each refresh starts a step inside every GPU job, so this panel is fetched only while
  its section is open.

![My Running Jobs](screenshots/my_jobs.png)

*Trimmed to the first seven jobs; ids, job names and the owner are placeholders.*

### Resource available

Live GPU, CPU and memory occupancy for every node, grouped by GPU type and sorted by
computing power, with a state badge per node (`idle`, `mix`, `alloc`, `drng`, `resv`)
and the users currently on it. Offline nodes are dimmed and counted separately.

![Resource available](screenshots/resource.png)

*Trimmed to the first four GPU types; usernames are replaced with placeholders.*

### GPU Leaderboard

GPUs held per user, in descending order. Each row lists the user's total, followed by the
number of 48 GB cards (`48g`), cards newer than the P40/M40 generation (`newer`), cards
held by interactive rather than batch jobs (`bash`), and the count per GPU type. Users
are shown as "Full Name (computing ID)", taken from the passwd entry. The row of the user
running `app.py` is highlighted and prefixed with `>`.

### Disk Quota

The output of `hdquota` as a table: size, used, available and percent used for your home,
scratch and research storage.

### Allocations

The picked account's service-unit allocations: allocated, remaining, percent used, and
which one is currently active. The `allocations` command reports a balance only to
members of the account, so for an account you do not belong to the panel says so instead
of showing an empty table.

![Allocations](screenshots/allocations.png)

### Queue Overview (All Users)

Cluster-wide contention per GPU type: GPUs free, jobs waiting, GPUs requested, and the
resulting queue depth. Jobs that could run on more than one GPU type are counted under
each, so the `Flexible` column overlaps between rows. Jobs held by a dependency or a
QOS/array limit are reported separately as `Not Competing` rather than inflating the
numbers.

![Queue Overview](screenshots/queue_overview.png)

### Estimated Wait for a Standard Job

Estimates when a job submitted now would start in each GPU pool. Enter a job shape (GPUs,
CPUs, memory, walltime; the account comes from the picker) and each row replays the
scheduler for that pool: every pending job that `sprio` ranks above a fresh submission of
yours is placed first, in priority order with backfill, as running jobs free their GPUs.

- `Worst Case` lets every job run to its time limit.
- `Typical` cuts each job to the fraction of its limit that jobs in that partition
  actually used over the last 3 days.

![Estimated Wait](screenshots/wait_estimate.png)

### Priority in Each Pool

Where a job submitted now under the picked account would rank among the jobs already
queued *in that partition*. Ranking against the whole cluster would be misleading,
because the partition factor shown next to each pool is added to every job in it alike.

![Priority](screenshots/priority.png)

### Waiting Queue

The picked account's pending jobs in `squeue`'s own columns: job id, partition, user,
time limit, nodes, submit time, start time and the reason each job is waiting.
`START_TIME` is the backfill scheduler's own estimate (what `squeue --start` reports) and
reads N/A for a job it has not planned, such as a held one.

## Command-Line Tool

The same GPU summary is available in the terminal. From the repository root:

```sh
python slurm_gpustat.py            # current usage
python available_resources.py      # available resources only
```

For convenience, add an alias pointing at your clone to `~/.bashrc`:

```sh
alias slurm_gpustat='python ~/rivanna_resource/slurm_gpustat.py'
```

## Changelog

- **2026-09-18** -- Faster refresh, about a third of the previous time: panels are fetched
  concurrently, all nodes are read in one `scontrol` call, and identical SLURM queries are
  shared. Own row highlighted in the leaderboard, stable ordering of tied GPU types, and
  access through Open OnDemand.
- **2026-09-17** -- Live per-job panel: CPU, RAM, GPU utilisation, GPU memory, power and
  temperature for your own running jobs, with GPU figures averaged over a few seconds of
  the driver's own samples.
- **2026-09-10** -- Header account picker; panels stay up when SLURM or the login node has
  a bad moment.
- **2026-09-03** -- Estimated wait time, queue overview, per-pool priority, collapsible
  sections, and the B200 / RTX PRO 6000 partitions.
- **2025-08-12** -- Multi-Instance GPU partition.
- **2025-04-29** -- Disk quota, H200 partition, and manual update button.
- **2024-10-31** -- Allocations panel; first customized release for UVA Rivanna.

## Acknowledgements

Based on [`slurm_gpustat`](https://github.com/albanie/slurm_gpustat) by
[Samuel Albanie](https://github.com/albanie) and `slurm_web` by
[Tengda Han](https://tengdahan.github.io/). Adapted and maintained for Rivanna by the UVA
Computer Vision Lab. Further documentation of the original command-line tool is available
in the [slurm_gpustat repository](https://github.com/albanie/slurm_gpustat).

## License

Released under the MIT License. See [LICENSE](LICENSE).
