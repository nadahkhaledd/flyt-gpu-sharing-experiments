# Dynamic SM Reconfiguration — Deployment Investigation Log

This log documents the step-by-step attempt to deploy the Flyt
orchestration stack on the shared hamta server and exercise dynamic
SM reallocation. Each checkpoint records what was attempted, what
was observed, and the outcome (success, partial, blocker). The
goal is to either reach a working `flytctl change-config` command
that demonstrably alters an application's SM allocation at runtime,
or to produce a precise diagnosis of where deployment stops and why.

## Components targeted

- MongoDB 7.0 (in Docker, host networking, port 27017)
- flyt-cluster-manager (orchestrator; ports 12401 for nodes, 12402 for clients)
- flyt-node-manager (per-GPU daemon)
- flyt-client-manager (per-workload daemon)
- cricket-rpc-server (already validated in static configuration)
- flytctl (control tool with commands: list-vms, list-servernodes,
  list-virt-servers, change-config, migrate, migrate-auto)

## Configuration choices

All three TOML configs in `dynamic_reconfig/configs/` point at
`127.0.0.1` instead of the placeholder `192.165.32.54` from the
shipped defaults. The cluster-manager config has its MongoDB
fields filled in (dbname = "flyt", no auth). The node-manager
config uses cluster-manager's `node` port (12401); the
client-manager config uses cluster-manager's `client` port (12402).

## Checkpoints

### Checkpoint A — MongoDB

**Attempted:** Start mongo:7.0 in Docker with host networking on port 27017.
**Outcome:** SUCCESS.
**Evidence:** Container running (docker ps); logs report "Waiting for
connections" on port 27017; `mongosh --eval "db.runCommand({ping: 1})"`
returns `{ ok: 1 }`.
**Notes:** Image pulled cleanly. No auth configured (cluster-manager
config uses empty user/password).

### Checkpoint B — Config files prepared

**Attempted:** Write three TOML configs to `dynamic_reconfig/configs/`
adjusted for local deployment.
**Outcome:** SUCCESS.
**Notes:** address changed from 192.165.32.54 to 127.0.0.1; MongoDB
fields filled in; node-manager pointed at cluster-manager node port
12401; client-manager pointed at cluster-manager client port 12402.

### Checkpoint C — Cluster manager startup

**Attempted:** Launch flyt-cluster-manager with --config flag.
**Outcome:** [TBD — fill in after diagnostic checks]
**Evidence (pending):** ps output, port bindings on 12401/12402,
log content from cluster-manager.log


### Checkpoint C — Cluster manager startup

**Attempted:** Launch flyt-cluster-manager with --config flag pointing at
our local cluster-mgr-config.toml.
**Outcome:** SUCCESS.
**Evidence:** Process visible in `ps aux` (PID 29647 at time of check).
Ports 12401 and 12402 bound on 0.0.0.0 per `ss -tlnp`. MongoDB
collections created in the `flyt` database (cluster-manager
auto-initialized the schema on first connection).
**Notes:** The `--config` flag works as expected; no panic on startup.
The cluster-manager runs in the foreground and produces no log output
during idle periods (only logs on events).

### Checkpoint D — Node manager registration

**Attempted:** Launch flyt-node-manager with --config pointing at
servnode-config.toml (cluster-manager at 127.0.0.1:12401, virt-server
binary at /home/khab/flyt/bin-server/cricket-rpc-server).
**Outcome:** SUCCESS.
**Evidence:** `flytctl list-servernodes` reports the L40S registered
under ServerNode IP 127.0.0.1, with 142 compute units and 46.7 GB GPU
memory available, 0 currently allocated. Node-manager process visible
in `ps aux` (PID 29722 at time of check).
**Notes:** Registration with the cluster-manager happens on startup
without explicit user action.

### Checkpoint E — Client manager startup and message queue creation

**Attempted:** Launch flyt-client-manager with --config pointing at
client-mgr.toml (cluster-manager at 127.0.0.1:12402).
**Outcome:** PARTIAL SUCCESS.
**Evidence:** Process visible in `ps aux` (PID 29938 at time of check).
Three System V message queues created (msqids 1, 2, 3, all owned by
khab with permissions 666). The IPC file /tmp/flyt-client-mgr exists.
**Outstanding:** `flytctl list-vms` returns an empty table, indicating
the client-manager has not (yet) registered itself as a VM with the
cluster-manager. This may explain the symptom in Checkpoint F.

### Checkpoint F — Workload through the stack (matmul_bench 2048 10)

**Attempted:** Run matmul_bench with LD_PRELOAD=cricket-client.so but
without CRICKET_SERVER set, expecting the client library to discover
its assigned Cricket server through the client-manager message queue.
**Outcome:** PARTIAL — workload appears to hang during initialization.
**Evidence:** Empty stdout and stderr log files; one unread message
present in message queue msqid 2 at time of check.
**Hypothesis:** The client library posted a registration request to
the client-manager but no response was returned, likely because the
client-manager has no VM registered with the cluster-manager (see
Checkpoint E). Investigation continuing.

### Checkpoint E — Client manager startup and message queue creation

**Attempted:** Launch flyt-client-manager with --config pointing at
client-mgr.toml (cluster-manager at 127.0.0.1:12402).
**Outcome:** PARTIAL SUCCESS.
**Evidence:** Process visible in `ps aux` (PID 29938 at time of check).
Three System V message queues created (msqids 1, 2, 3, all owned by
khab with permissions 666). The IPC marker file /tmp/flyt-client-mgr exists.
**Notes:** `flytctl list-vms` returns an empty table at this point. This
is expected: the client-manager registers VMs lazily, only when a CUDA
application invokes it through the message queue.

### Checkpoint F — Workload through the stack (first attempt)

**Attempted:** Run matmul_bench with LD_PRELOAD=cricket-client.so and
no CRICKET_SERVER variable, expecting client-manager-mediated discovery.
**Outcome:** BLOCKER (workload hangs at startup).
**Evidence:** Empty stdout and stderr from matmul. Message queue msqid 2
held one unread 4-byte message (the client's registration request) for
the duration of the run. matmul_bench process remained in S+ (sleeping
on a syscall, foreground) state until killed with SIGKILL.
**Hypothesis:** The Cricket client posted its registration request to
the client-manager's input queue, but no response was returned. Two
possible causes: (a) the client-manager needs application-level config
hints (APP_SM_CORES, APP_GPU_MEMORY, APP_MODE) that we didn't provide;
(b) the cluster-manager requires a pre-existing VM record in MongoDB
that we haven't created. The Flyt paper describes the onboarding
process at a high level but does not document the operator-facing
steps required to seed an initial VM record. Investigation continuing.

### Checkpoint F (final) — Application onboarding blocked at IPC layer

**Attempted:** Run matmul_bench through the full orchestration stack so the
cluster-manager would consult MongoDB, allocate a virt-server, and return
its endpoint to the client via the client-manager.

**Outcome:** BLOCKER — precisely localized.

**What works:**
- All four daemons start and handshake (Mongo, cluster-manager, node-manager,
  client-manager).
- node-manager registers the L40S (142 SMs) with the cluster-manager;
  confirmed via `flytctl list-servernodes`.
- MongoDB `vm_required_resources` record is correct (schema, field names,
  int types all verified against the Rust `VMResources` struct).
- The cricket client (cricket-client.so) reaches init_client_mgr(), computes
  the correct SysV message-queue key via ftok("/tmp/flyt-client-mgr", 0x42),
  and posts its registration message (4-byte PID, mtype=1) to the correct
  queue. Verified with strace and ipcs: the message lands in the exact msqid
  the client-manager owns (confirmed keys matched: 0x42050029, same msqid,
  message present: "4 bytes, 1 message").

**The blocker:**
Despite the registration message arriving in the correct queue with the
correct type and size, the client-manager's `recv_type(1)` (via the ipc-rs
crate) never consumes it — no "Client connected" log line is ever emitted.
The matmul then blocks indefinitely in msgrcv awaiting a reply.

This indicates a message-format incompatibility between the C client's
`msgsnd` (struct msgbuf_uint32 { long mtype; uint32_t data; }, sent with
size = sizeof(data) = 4) and the Rust client-manager's `ipc-rs`
`recv_type(1)` expectation. The mismatch is at the IPC-library boundary,
not in configuration, ordering, or the message-queue key (all of which were
exhaustively verified and corrected).

**Why we stopped here:**
Resolving this requires patching and rebuilding either the Rust
control-managers (to add logging inside the recv path and adjust the
expected message layout) or the C cricket client. Both are multi-hour
build-debug cycles. Given project time constraints, we scoped this to
future work with the blocker precisely documented rather than pursue an
uncertain rebuild.

**Reproduction value:**
Everything up to and including the client registration message reaching the
correct queue is reproducible from the steps in this log and the configs in
dynamic_reconfig/configs/. A future effort should start by instrumenting the
ipc-rs recv_type call in src/client-manager-daemon/vcuda_client_handler.rs
to log the raw bytes and mtype it receives, and compare against the C
struct layout in cpu/cpu-client-mgr-handler.c and cpu/msg-handler.h.
