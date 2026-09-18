# Docker-backed evals: sandbox & network setup

Several suites run each task inside its own Docker container (and, for compose-based ones,
its own Docker **network**):

- `terminal_bench`: Harbor spins up one Docker Compose environment per trial
- `swe_bench_pro`, `swe_bench_live`, `swe_lancer`, `multi_swe_bench`: one container per
  instance, applying the model's patch and running the repo's tests

At `--sandboxes N` these run up to N containers/networks **concurrently**. That concurrency
is what makes them fast, and it is also what exhausts two finite Docker resources if the
daemon is left on its defaults: the **address pool** and **stale containers**.

## The failure this prevents

On a default daemon you will eventually hit, mid-run:

```
Error response from daemon: all predefined address pools have been fully subnetted
failed to create network <task>__env_default
```

Every affected task then dies with a bare `RuntimeError`, an empty transcript, and a score
of 0, indistinguishable from a model failure unless you read the trial logs. Measured on a
`terminal_bench` run at `--sandboxes 32`: **69 of 89 tasks** failed this way and the suite
reported 6.74% (a floor, not a measurement).

Docker's default `default-address-pools` yield only ~31 networks total
(`172.17.0.0/12` size 16 **+** `192.168.0.0/16` size 20). One compose network per concurrent
trial, plus the default bridge, plus any leftover containers, overruns that almost
immediately.

## Recommended daemon configuration

Give Docker a large, **bounded** pool carved into small subnets, in a range you have
confirmed is free on the host. This is a best practice for any box that runs these suites at
concurrency.

```json
{
  "default-address-pools": [
    { "base": "172.28.0.0/14", "size": 24 }
  ]
}
```

`172.28.0.0/14` size 24 = 2^(24−14) = **1024 networks**, far more than any `--sandboxes`
value needs, while leaving the rest of the RFC-1918 space free for other workloads and
manually-created networks.

Apply it (requires root; the restart kills all running containers, so do it while nothing
important is running):

```bash
sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
{
  "default-address-pools": [
    { "base": "172.28.0.0/14", "size": 24 }
  ]
}
EOF
sudo systemctl restart docker
docker system info --format '{{json .DefaultAddressPools}}'   # verify
```

### Pick a range that does not collide with your host

**Do not blindly copy the base above.** Before applying, confirm the range is unused on
*your* host; a pool that overlaps the host's own NIC, a VPN, or a corporate subnet will
break connectivity, not just Docker.

```bash
ip -o -4 addr show | awk '{print $2, $4}'        # host interfaces
docker network ls -q | xargs -r docker network inspect \
    -f '{{.Name}} {{range .IPAM.Config}}{{.Subnet}}{{end}}'   # subnets already in use
ip route | grep -E '172\.(2[89]|3[01])\.'        # anything in 172.28-172.31? (empty = clear)
```

In particular, **`192.168.0.0/16` is a common host/LAN range**; Docker's *default* pool
includes it, so on a box whose own address is `192.168.x.y` the stock daemon can already
hand out a colliding network. The recommended config deliberately omits `192.168/16` for
this reason. If `172.28.0.0/14` is not free on your host, pick another cleanly-unused block
(e.g. a slice of `10.0.0.0/8`, checked the same way) and keep `size: 24`.

## Also: prune before a run

The pool holds live *and* recently-exited containers' addresses. Clear stale state before a
sweep (safe when nothing you need is running):

```bash
docker container prune -f && docker network prune -f
```

Do this **before** the sweep, never during; pruning races networks your own in-flight tasks
are still attaching.

## Concurrency without the pool fix

If you cannot edit the daemon (no root), keep `--sandboxes` at or below ~12-16 for these
suites so concurrent networks stay under the ~31 default limit, and prune first. It is
slower, but it avoids the silent 0-score failures above.

---
For the `/tmp` inode limit (the other resource a big sweep exhausts) and a full pre-sweep checklist, see [large-sweep-setup.md](large-sweep-setup.md).
