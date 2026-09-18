# Panels

[← Back to the README](../readme.md)

Panels are listed in page order. **Scope** says whose data it is: the whole *cluster*,
the *account* picked in the header, or *you*, meaning whoever runs `app.py`.

Everything loads when you open the page, when you come back to its tab, and when you
press *Update All*. Nothing polls in the background.

## My Running Jobs (live)

*Scope: you · Built from `squeue`, `sstat`, `srun`*

What your jobs are really using: busy cores, memory, and per GPU its utilisation, memory,
power and temperature.

![My Running Jobs](../screenshots/my_jobs.png)

- Loads only while the section is open, because each refresh runs a short step inside
  every GPU job (`srun --overlap`).
- One GPU reading swings between 45% and 100% on a steady job, so utilisation and power
  are averaged over samples the driver already buffered (about 5 s and 2.4 s), at no
  extra wait. Hover a bar to see the window.

## Resource available

*Scope: cluster · Built from `sinfo`, `squeue`, `scontrol`*

GPU, CPU and memory of every node, grouped by GPU type, with a state badge (`idle`, `mix`,
`alloc`, `drng`, `resv`) and who is on it. Offline nodes are dimmed.

![Resource available](../screenshots/resource.png)

## GPU Leaderboard

*Scope: cluster · Built from `sinfo`, `squeue`*

Who holds how many GPUs, most first. Your own row is highlighted and marked with `>`.

- `48g`: 48 GB cards · `newer`: anything past the P40/M40 generation · `bash`: held by
  interactive rather than batch jobs.
- Names come from the passwd entry: "Full Name (computing ID)".

## Disk Quota

*Scope: you · Built from `hdquota`*

Size, used and available for your home, scratch and research storage.

## Allocations

*Scope: account · Built from `allocations`*

Service units: allocated, remaining, percent used, and which allocation is active. You
only see balances of accounts you belong to.

![Allocations](../screenshots/allocations.png)

## Queue Overview (All Users)

*Scope: cluster · Built from `sinfo`, `squeue`, `scontrol`*

Per GPU type: GPUs free, jobs waiting, GPUs they want, and queue depth.

![Queue Overview](../screenshots/queue_overview.png)

- A job that fits several GPU types is counted under each, so `Flexible` overlaps.
- Jobs blocked by a dependency or a QOS/array limit are shown as `Not Competing`.

## Estimated Wait for a Standard Job

*Scope: account · Built from `squeue`, `sprio`, `sshare`, `sacct`, `sacctmgr`, `scontrol`,
`sinfo`, `sbatch --test-only`*

"If I submit now, when do I start?" Enter GPUs, CPUs, memory and walltime; each row
replays the scheduler for one GPU pool, placing every job ranked above yours first.

![Estimated Wait](../screenshots/wait_estimate.png)

- `Worst Case`: every job runs to its time limit.
- `Typical`: every job runs for the fraction of its limit that jobs in that partition
  really used over the last 3 days.

## Priority in Each Pool

*Scope: account · Built from `sprio`, `sshare`, `sacctmgr`, `scontrol`, `sinfo`*

Where a new job from your account would rank among the jobs queued in each partition.
Ranks are per partition because the partition factor is added to every job in it alike.

![Priority](../screenshots/priority.png)

## Waiting Queue

*Scope: account · Built from `squeue`*

Your account's pending jobs with `squeue`'s own columns. `START_TIME` is SLURM's own
estimate and reads N/A for jobs it has not planned, such as held ones.
