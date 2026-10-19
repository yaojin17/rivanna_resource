# Rivanna Monitor

One page that shows what UVA's Rivanna cluster is doing, where your account stands in the
queue, and what your own jobs are using.

![Resource available panel](screenshots/resource.png)

## Quick start

On a Rivanna login node:

```sh
git clone https://github.com/UVA-Computer-Vision-Lab/rivanna_resource.git
cd rivanna_resource
pip install -r requirements.txt
python app.py
```

Then open it in your browser through Open OnDemand. No SSH tunnel, just your NetBadge
login:

```
https://ood.hpc.virginia.edu/rnode/<node>/2070/
```

`<node>` is what `hostname` prints on the node where you started it.

**Don't want to start it by hand every time?** A small login hook keeps it running and
prints the URL for you. See [Deployment](docs/deployment.md).

## What's on the page

| Panel | What it tells you |
| --- | --- |
| **My Running Jobs** | CPU, memory, GPU utilisation, power and temperature of your own jobs, live |
| **Resource available** | Every node: GPUs, CPUs, memory, and who is on it |
| **GPU Leaderboard** | Who holds how many GPUs, with your row highlighted |
| **Disk Quota** | Your home, scratch and research storage |
| **Allocations** | Your account's service units |
| **Queue Overview** | How contended each GPU type is |
| **Estimated Wait** | Modelled waits for a new job by you under the selected account |
| **Priority** | Your own priority percentile among competing GPU tasks in each pool |
| **Running Jobs** | Your lab's running jobs and the GPUs they hold |
| **Waiting Queue** | Your lab's pending jobs |

Pick your account in the header and the account panels follow it.
Screenshots and details: [Panels](docs/panels.md).

## Settings

```sh
RIVANNA_ACCOUNTS=my_lab,other_lab python app.py    # accounts offered in the header
```

All settings: [Deployment › Settings](docs/deployment.md#settings).

## In the terminal

```sh
python slurm_gpustat.py          # GPU usage across the cluster
python available_resources.py    # only what is free
```

## What's new

- **2026-10-19** Priority and wait estimates use the monitor owner's fair-share
  and submission QoS under the selected account, or its members' median for an
  account the owner is not in. Priority counts every array task.
  Queue Overview and the replay check feature constraints and requested or excluded
  nodes. Job tables preserve array indices and minute-level time limits. Memory
  inputs use GiB, matching `sbatch --mem=NG`. The wait columns now say Full Limits,
  Historical Runtime and Model Start; they are estimates, not scheduler guarantees.
- **2026-10-15** Running Jobs shows the GPU type each job landed on. Free GPUs were
  overstated and are now counted right on the page and in both terminal tools: GPUs
  held through the mixed `gpu` and `interactive` partitions count as in use, and nodes
  reserved for other groups, not responding (`comp*`) or stuck cleaning up after a job
  (`comp`) count as offline. Queue Overview also counts jobs that ask with `--gpus`,
  and it, Waiting Queue and Estimated Wait count every task of a long pending array
  (one had counted 145 of 433). Estimated Wait is about twice as fast from cold and
  blanks only its Typical column when `sacct` fails. The app no longer needs django,
  humanize or humanfriendly.
- **2026-09-30** Waiting Queue ends with a total of pending jobs and requested GPUs,
  and a pending job array's row shows its task range.
- **2026-09-28** Running Jobs panel for your account. The leaderboard no longer counts
  jobs that are completing, and now counts jobs that ask for GPUs with `--gpus`.
- **2026-09-24** Estimated Wait reuses the last start of repeated array tasks and
  searches the earliest part of the 30-day window first, speeding up large queues.
  Waiting Queue shows requested GPUs per job instead of node count.
- **2026-09-18** Refreshes about 3× faster, your leaderboard row is highlighted, and the
  page works through Open OnDemand.
- **2026-09-17** Live CPU, memory and GPU panel for your own running jobs.
- **2026-09-10** Account picker; panels stay up when SLURM has a bad moment.
- **2026-09-03** Wait estimate, queue overview, per-pool priority, B200 and RTX PRO 6000.
- **2025-08-12** Multi-Instance GPU partition.
- **2025-04-29** Disk quota, H200 partition, manual update button.
- **2024-10-31** Allocations; first release for Rivanna.

## Troubleshooting

| Problem | Root cause | Solution and lesson |
| --- | --- | --- |
| Priority looks favourable while another member of the same lab gets the GPUs | The old profile used the median fair-share of all account members. It also counted each compressed array as one competing job. | Use the server owner's exact user/account association and count all pending tasks. On 2026-10-19, the median was 0.025206 while rry4fg had 0.024794. Waiting age can decide the order even within one lab. |
| H200 appears to have demand from jobs that exclude H200 | The old candidate calculation used partition names and typed GRES, ignoring Features and node lists. | Apply the live feature alternatives and requested/excluded nodes before counting candidates or placing work in the replay. Read squeue fields with tab separators because feature alternatives contain pipes. |
| A label job shows 2h, or an array task has an unfamiliar numeric ID | The display rounded limits to hours and exposed Slurm's internal row ID. | Keep minutes and show the array parent plus task index, such as `21524330_44`, with its actual `1h30m` limit. |
| A 96G job is estimated with less memory than its actual request | The input multiplied by 1000, but Slurm uses 1024 MiB per GiB. | Convert GiB to MiB with 1024 and show the unit in the form. Free GPUs alone do not establish that the CPU, memory and duration fit. |
| Estimated Wait appears to promise a start time | The old Worst Case and Typical labels overstated a model with incomplete scheduler information. | Full Limits and Historical Runtime describe assumptions. Future reservations, arrivals, priority changes and user limits can move the result. Use Waiting Queue for Slurm's forecast of an already submitted job. |
| *Update All* spends several seconds at 8/9 while Estimated Wait loads | The scheduler replay expanded a large pending job array and searched the full 30-day window for every copy, even when a later copy could not start before an earlier one. | The replay now starts each repeated task at the previous copy's earliest start, searches time in growing chunks, and stops retrying copies after one cannot fit. If this checkout was updated while the app was running, restart `python app.py` to load the change. Profile the replay before blaming `sbatch --test-only`; its checks were short in the 2026-09-24 investigation. |

On 2026-09-24, a pending array of about 1,523 three-day tasks made this panel the
slowest. The revised replay matched the old start times for every pool on the same live
SLURM snapshot. In a nine-panel HTTP refresh, Estimated Wait fell from 7.58 s to 3.39 s;
the exact time depends on the current queue.

The 2026-10-19 audit used the live queue on `udc-ba06-6c0:2070`. All 56 accessible
GPU inventory nodes matched Slurm's allocated GPU counts. Disk Quota and Allocations
matched their command output. Fourteen HTTP checks passed after restarting that exact
instance: all ten data panels, the index and clock, a second account's priority, and a
1 GPU / 4 CPU / 96 GiB / 8h estimate. Real pending rows verified the personal fair-share,
feature exclusions, array indices and 90-minute labels. The dedicated percentile for a
fresh rry4fg job changed from about 99% to 0%; it is not the rank of an existing aged job.
The queue replay remains a model, particularly when flexible tasks share several pools.

## Credits

Built on [`slurm_gpustat`](https://github.com/albanie/slurm_gpustat) by Samuel Albanie and
`slurm_web` by [Tengda Han](https://tengdahan.github.io/), written for the
[Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/), Oxford. Maintained by the UVA
Computer Vision Lab. [MIT licensed](LICENSE).
