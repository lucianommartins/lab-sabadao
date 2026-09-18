# Host setup for a large sweep

Before running many evals at high concurrency (`--batch-sizes`/`--sandboxes` in the
hundreds/dozens), prepare two host resources that a big run silently exhausts. Both failures
present as misleading errors and score healthy work as 0, so set them up **before** you
launch a multi-hour sweep.

## 1. Raise the `/tmp` inode limit (tmpfs)

On most machines `/tmp` is a **tmpfs** with a fixed inode cap (`nr_inodes`, commonly ~1M).
That cap is on the *number of files*, not bytes. Several suites create large numbers of
small files under `/tmp`:

- **tau2 / tau3** stage the retrieval **knowledge base** (~700 docs) into
  `/tmp/agentic_search_<id>/` per simulation. Its `SandboxManager` cleans up on graceful
  exit, but a simulation that is killed or times out (e.g. a context-window overflow) leaves
  its directory behind. Under concurrency these **leak and accumulate across runs**.
- **aider_polyglot** copies each exercise's test files into a `/tmp` scratch dir per task.
- bubblewrap-sandboxed suites each get a `--tmpfs /tmp` per invocation.

When the inode table fills, the next file creation fails with `ENOSPC`
(`[Errno 28] No space left on device`) **even though `df -h` shows the disk almost empty**,
because it is *inodes*, not space. Measured case: a run accumulated ~1,141 leaked
`agentic_search_*` dirs (~800K inodes, 91% of a 1,048,576 cap); the next sweep's
aider_polyglot hit the cap and was recorded `status: error, N=0`, a whole eval lost to a
misread error.

**Recommended before a big sweep** (live remount, no reboot, no data loss; requires root):

```bash
sudo mount -o remount,rw,nosuid,nodev,nr_inodes=0,inode64 /tmp
df -i /tmp    # IFree should now be very large
```

`nr_inodes=0` removes the inode cap (bounded only by the tmpfs size / RAM). The options are
copied from the current mount so nothing else changes. Check yours first with
`findmnt /tmp -o OPTIONS` and preserve them. Prefer a finite bump (e.g.
`nr_inodes=8388608`) if you want a ceiling.

Diagnose an inode problem with:

```bash
df -i /tmp                              # IUse% near 100 = inode exhaustion
du --inodes -d1 /tmp | sort -rn | head # find the hoarder
```

### After the run: reclaim the leaked dirs

Raising the cap is a band-aid; the tau leftovers persist (RAM-backed) until removed. When no
sweep is running:

```bash
rm -rf /tmp/agentic_search_*
```

## 2. Widen the Docker address pool

Container-backed suites (`terminal_bench`, `swe_bench_pro`, `swe_bench_live`, `swe_lancer`,
`multi_swe_bench`) create one Docker network per concurrent task and exhaust the default
address pool (~31 networks) almost immediately at high `--sandboxes`. See
[docker-sandboxes.md](docker-sandboxes.md) for the recommended `default-address-pools`
config and the pre-run `docker … prune`.

## Quick pre-sweep checklist

```bash
sudo mount -o remount,rw,nosuid,nodev,nr_inodes=0,inode64 /tmp   # 1. inodes
# 2. Docker pool: see docker-sandboxes.md (one-time daemon.json + restart)
docker container prune -f && docker network prune -f            # clear stale containers/nets
rm -rf /tmp/agentic_search_*                                     # clear stale tau sandboxes
df -i /tmp                                                       # confirm inode headroom
```
