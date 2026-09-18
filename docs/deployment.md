# Deployment

[← Back to the README](../readme.md)

- [Open it in your browser](#open-it-in-your-browser)
- [Keep it running](#keep-it-running)
- [Troubleshooting](#troubleshooting)
- [Settings](#settings)

## Open it in your browser

Login nodes have no public address, so pick one of these.

### Open OnDemand (easiest)

```
https://ood.hpc.virginia.edu/rnode/<node>/<port>/
```

- `<node>` is what `hostname` prints where the monitor runs. `<port>` is 2070 unless you
  changed it. The page title shows both.
- It is `rnode`, not `node`. `/node/` gives *Not Found* on Rivanna.

### SSH tunnel

```sh
ssh -L 2070:<node>:2070 <computing-id>@login.hpc.virginia.edu
```

Then open `http://localhost:2070`. Keep the node name in the command:
`login.hpc.virginia.edu` hands you a login node at random, which may not be the one
running the monitor.

## Keep it running

`python app.py` stops when its terminal closes, and Rivanna does not allow `crontab`, `at`
or lingering `systemd --user` services. A login hook does the job instead.

**1. Pick a port.** If labmates share your login node, each person needs their own.
Anything free above 1024 works.

**2. Save this as `~/.config/rivmon.sh`** and set the first two lines:

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

**3. Load it from `~/.bashrc`**, and from `.zshrc` too if you use zsh:

```sh
[ -f "$HOME/.config/rivmon.sh" ] && . "$HOME/.config/rivmon.sh"
```

That's it. Each new shell starts the monitor if it is not running and prints its URL.
Type `rivmon` to see the URL again.

### Good to know

- Logging out does not stop it. After a node reboot, your next login brings it back.
- Login nodes rotate, so each node you use runs its own copy, with its own URL.
- If someone else holds your port, the hook says so instead of showing you their URL.
- `scp`, `rsync` and `ssh <host> <command>` are untouched; the hook only runs in
  interactive shells.
- Logs go to `app.log` in the repository. The `python` on your `PATH` needs the
  requirements installed.

## Troubleshooting

| You see | Why | Do this |
| --- | --- | --- |
| *Not Found* for the whole page | The URL says `/node/` | Use `/rnode/<node>/<port>/` |
| An OnDemand error instead of the page | Nothing is listening on that node and port | Run `rivmon` on that node |
| The page loads, every panel says *Not Found* | Your checkout predates 2026-09-18 | `git pull` |
| "port ... is taken by someone else" | A labmate uses that port on this node | Change `RIVMON_PORT` |
| Yesterday's URL is dead | You landed on a different login node | Run `rivmon` for today's URL |
| Gone after maintenance | The node rebooted | Log in to it once; the hook restarts it |

## Settings

Environment variables, read when `app.py` starts.

| Variable | Default | What it does |
| --- | --- | --- |
| `RIVANNA_ACCOUNTS` | `uva_cv_lab,cang-lab-in-silico` | Accounts in the header picker, comma separated. First one is the default. |
| `RIVANNA_STANDARD_HOURS` | `24` | Walltime the wait-estimate form starts with. |
| `RIVANNA_WALLTIME_DAYS` | `3` | Days of finished jobs used for the `Typical` wait. |
| `RIVANNA_WALLTIME_TTL` | `1800` | Seconds before that sample is refreshed. |
| `RIVANNA_TEST_ONLY_TTL` | `60` | Seconds an `sbatch --test-only` verdict is kept. |
| `RIVANNA_CMD_CACHE_SECONDS` | `10` | Seconds identical SLURM queries are shared between panels. `0` turns it off. |

`--host` and `--port` set where it listens (default `0.0.0.0:2070`). Header, footer and
styling live in [index.html](../index.html).
