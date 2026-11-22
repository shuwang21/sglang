#!/usr/bin/env bash
# One-shot gpt-oss loop session for short-lived GPU allocations.
#
# With 2 or more GPUs, the BASELINE and LOOP arms run AT THE SAME TIME: the
# first half of the GPUs serve the baseline, the second half the loop, one
# server per GPU, and each arm's eval spreads its requests over its servers.
# With 1 GPU the arms run one after the other (baseline, then loop).
#
# The eval runs (question, seed) pairs in a fixed hashed order, so a run that
# is cut off (TIME_BUDGET_MIN, Ctrl-C, allocation end) still holds a uniform
# random subset, and the paired report covers what both arms finished. The
# report and the tarball are refreshed every REPORT_EVERY_MIN minutes.
#
# Usage (from the repo root, venv active):
#   TIME_BUDGET_MIN=200 bash benchmark/looped_transformer/run_session.sh
#
# Everything worth keeping is under results/gptoss_sessions/session_<timestamp>/
# (also packed into session_<timestamp>.tar.gz). Pull it to the laptop with
# fetch_results.sh before the allocation ends.
#
# Knobs (environment variables, all optional):
#   MODEL           default openai/gpt-oss-20b
#   LOOP_CONFIG     default: the 12-15 / k=2 / beta=0.35 config
#   GPUS            comma-separated GPU ids; default: $CUDA_VISIBLE_DEVICES,
#                   else every GPU nvidia-smi lists
#   ATTENTION_BACKEND  server --attention-backend; default auto: fa3 on
#                   SM8x/SM9x (A100, H100, H200), triton elsewhere (B200 is
#                   SM100, where fa3 refuses to start). triton supports the
#                   sinks and sliding window gpt-oss needs and honours the
#                   loop's save_kv_cache=False.
#   PORT            first server port, default 30000 (server i uses PORT+i)
#   CONCURRENCY     in-flight requests per server, default 64 (a task's own
#                   --concurrency overrides it)
#   TIME_BUDGET_MIN stop the evals this many minutes after the session starts
#                   and write the final report; 0 (default) = no limit. With 1
#                   GPU the baseline gets half of what is left when it starts.
#   REPORT_EVERY_MIN  refresh reports + tarball while evals run, default 10
#   PYTHON          default python
#   SESSION_DIR     default results/gptoss_sessions/session_<timestamp>;
#                   point it at an existing session to resume it
#   TASKS_FILE      file with one task per line: "<name>|<eval_gptoss run args>"
#                   (lines starting with # are ignored); default: TASKS below
#   EXTRA_SERVER_ARGS  appended to BOTH server commands
#   STARTUP_TIMEOUT seconds to wait for /health, default 1200
#   PUSH_RESULTS    0 (default): the shared GPU nodes forbid leaving any
#                   GitHub / Hugging Face credentials on the box, so results are
#                   pulled from the laptop with fetch_results.sh instead.
#                   1: commit+push the session dir from the node after every
#                   report (only where pushing needs no stored token).
#   UPLOAD_CMD      optional command run after every repack; "{}" is replaced
#                   by the tarball path, e.g. 'rclone copy {} myremote:gptoss/'

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

MODEL="${MODEL:-openai/gpt-oss-20b}"
DEFAULT_LOOP_CONFIG='{"start":12,"end":15,"k":2,"mode":"block","strategy":"block_anchored","anchor_beta":0.35,"cache_strategy":"first","decode_mode":"full"}'
LOOP_CONFIG="${LOOP_CONFIG:-$DEFAULT_LOOP_CONFIG}"
PORT="${PORT:-30000}"
PYTHON="${PYTHON:-python}"
CONCURRENCY="${CONCURRENCY:-64}"
TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-0}"
REPORT_EVERY_MIN="${REPORT_EVERY_MIN:-10}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
UPLOAD_CMD="${UPLOAD_CMD:-}"
PUSH_RESULTS="${PUSH_RESULTS:-0}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${GPUS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPUS="$CUDA_VISIBLE_DEVICES"
  else
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, - || true)"
    GPUS="${GPUS:-0}"
  fi
fi
IFS=, read -r -a GPU_LIST <<< "${GPUS// /}"
if (( ${#GPU_LIST[@]} >= 2 )); then
  half=$(( ${#GPU_LIST[@]} / 2 ))
  BASE_GPUS=("${GPU_LIST[@]:0:half}")
  LOOP_GPUS=("${GPU_LIST[@]:half:half}")
  PARALLEL=1
else
  BASE_GPUS=("${GPU_LIST[0]}")
  LOOP_GPUS=("${GPU_LIST[0]}")
  PARALLEL=0
fi

ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
if [[ "$ATTENTION_BACKEND" == auto ]]; then
  GPU_CAP="$(nvidia-smi -i "${GPU_LIST[0]}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ' || true)"
  case "$GPU_CAP" in
    8.*|9.*|"") ATTENTION_BACKEND=fa3 ;;  # "" = no nvidia-smi: keep the old default
    *) ATTENTION_BACKEND=triton ;;
  esac
fi

# Default plan for one 4h allocation: medium effort on ALL of MMLU-Pro (12k
# questions), seed 0. It will not finish in 4h; run it with TIME_BUDGET_MIN
# and the hashed order makes whatever finished a random subset. With 1 GPU,
# a stratified 36-per-category subset (~500 questions) instead.
# MMLU-Pro and gpt-oss-20b are not gated, so no Hugging Face login is needed.
# GPQA is gated: copy a local file to the node and add a task like
#   gpqa_diamond_medium|--dataset gpqa --data-file gpqa_diamond.csv --reasoning-effort medium --seeds 0 1
if (( PARALLEL )); then
  TASKS=("mmlu_pro_medium_full|--dataset mmlu_pro --reasoning-effort medium --seeds 0")
else
  TASKS=("mmlu_pro_medium_strat36|--dataset mmlu_pro --reasoning-effort medium --per-category 36 --seeds 0 1")
fi
if [[ -n "${TASKS_FILE:-}" ]]; then
  TASKS=()
  while IFS= read -r line; do
    [[ -z "${line// }" || "$line" == \#* ]] && continue
    TASKS+=("$line")
  done < "$TASKS_FILE"
fi

SESSION_DIR="${SESSION_DIR:-results/gptoss_sessions/session_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$SESSION_DIR"/{runs,reports,servers}
SESSION_DIR="$(cd "$SESSION_DIR" && pwd)"
TARBALL="$SESSION_DIR.tar.gz"
exec > >(tee -a "$SESSION_DIR/session.log") 2>&1

EVAL="$REPO/benchmark/looped_transformer/eval_gptoss.py"
SERVER_PIDS=()
SERVER_PORTS=()
SERVER_LOGS=()
EVAL_PIDS=()
T0=$(date +%s)
DEADLINE=0
(( TIME_BUDGET_MIN > 0 )) && DEADLINE=$(( T0 + TIME_BUDGET_MIN * 60 ))

log() { printf '[%s +%dm] %s\n' "$(date +%H:%M:%S)" $(( ($(date +%s) - T0) / 60 )) "$*"; }

pack() {
  # Re-pack the whole session (small: jsonl + logs + reports) into one file.
  tar czf "$TARBALL.tmp" -C "$(dirname "$SESSION_DIR")" "$(basename "$SESSION_DIR")" \
    && mv "$TARBALL.tmp" "$TARBALL"
  log "packed -> $TARBALL ($(du -h "$TARBALL" | cut -f1))"
  if [[ "$PUSH_RESULTS" == 1 ]]; then
    bash "$REPO/benchmark/looped_transformer/push_results.sh" "$SESSION_DIR" \
      || log "WARNING: git push of results failed (tarball is still local)"
  fi
  if [[ -n "$UPLOAD_CMD" ]]; then
    local cmd="${UPLOAD_CMD//\{\}/$TARBALL}"
    log "upload: $cmd"
    bash -c "$cmd" || log "WARNING: upload command failed"
  fi
}

report() {  # $1 = task name; paired report from whatever both arms have so far
  local name="$1" run="$SESSION_DIR/runs/$1"
  [[ -s "$run/baseline/samples.jsonl" && -s "$run/loop/samples.jsonl" ]] || return 0
  "$PYTHON" "$EVAL" summarize "$run/baseline" "$run/loop" \
    --output "$SESSION_DIR/reports/$name.md" --title "$name" > /dev/null \
    && log "report: reports/$name.md" \
    || log "WARNING: summarize $name failed"
  cat "$SESSION_DIR"/reports/*.md > "$SESSION_DIR/SUMMARY.md" 2>/dev/null || true
}

port_free() {  # $1 = port
  "$PYTHON" - "$1" <<'EOF'
import socket, sys
s = socket.socket()
# Like the server (uvicorn sets SO_REUSEADDR): TIME_WAIT leftovers from client
# connections do not block a new listener, only a live one does.
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
EOF
}

healthy() {  # $1 = port
  "$PYTHON" - "$1" <<'EOF'
import sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
EOF
}

alive() {  # running and not a zombie
  local state
  state=$(ps -o stat= -p "$1" 2>/dev/null || true)
  [[ -n "$state" && "$state" != Z* ]]
}

stop_servers() {
  (( ${#SERVER_PIDS[@]} )) || return 0
  local pid port
  log "stopping ${#SERVER_PIDS[@]} server(s) (pids ${SERVER_PIDS[*]})"
  for pid in "${SERVER_PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for _ in $(seq 60); do
    local any=0
    for pid in "${SERVER_PIDS[@]}"; do alive "$pid" && any=1; done
    (( any )) || break
    sleep 1
  done
  for pid in "${SERVER_PIDS[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  for port in "${SERVER_PORTS[@]}"; do
    local freed=0
    for _ in $(seq 60); do port_free "$port" && { freed=1; break; }; sleep 1; done
    (( freed )) || log "WARNING: port $port still busy after 60s"
  done
  SERVER_PIDS=(); SERVER_PORTS=(); SERVER_LOGS=()
}

start_server() {  # $1 = baseline|loop, $2 = gpu, $3 = port, $4 = log file
  local mode="$1" gpu="$2" port="$3" logfile="$4"
  local cmd=("$PYTHON" -m sglang.launch_server --model-path "$MODEL"
    --host 127.0.0.1 --port "$port"
    --tp-size 1 --attention-backend "$ATTENTION_BACKEND" --disable-radix-cache
    --disable-cuda-graph --log-level info --reasoning-parser gpt-oss)
  [[ "$mode" == loop ]] && cmd+=(--loop-config "$LOOP_CONFIG")
  # shellcheck disable=SC2206
  [[ -n "$EXTRA_SERVER_ARGS" ]] && cmd+=($EXTRA_SERVER_ARGS)

  for _ in $(seq 30); do port_free "$port" && break; sleep 2; done
  { echo "CUDA_VISIBLE_DEVICES=$gpu"; printf '%q ' "${cmd[@]}"; echo; } > "$logfile.cmd"
  log "starting $mode server on GPU $gpu, port $port -> $logfile"
  CUDA_VISIBLE_DEVICES="$gpu" setsid "${cmd[@]}" > "$logfile" 2>&1 < /dev/null &
  SERVER_PIDS+=("$!"); SERVER_PORTS+=("$port"); SERVER_LOGS+=("$logfile")
}

wait_servers() {  # every started server healthy, or fail
  local waited=0 i
  while :; do
    local ready=0
    for i in "${!SERVER_PIDS[@]}"; do
      if healthy "${SERVER_PORTS[$i]}"; then
        ready=$((ready + 1))
      elif ! alive "${SERVER_PIDS[$i]}"; then
        log "ERROR: server on port ${SERVER_PORTS[$i]} exited during startup; last lines of ${SERVER_LOGS[$i]}:"
        tail -n 40 "${SERVER_LOGS[$i]}"
        stop_servers
        return 1
      fi
    done
    (( ready == ${#SERVER_PIDS[@]} )) && break
    if (( waited >= STARTUP_TIMEOUT )); then
      log "ERROR: servers not healthy after ${STARTUP_TIMEOUT}s"
      stop_servers
      return 1
    fi
    sleep 5; waited=$((waited + 5))
  done
  log "${#SERVER_PIDS[@]} server(s) ready after ${waited}s"
}

start_arm() {  # $1 = mode, $2 = task name, $3 = eval args, rest = ports
  local mode="$1" name="$2" args="$3"; shift 3
  local urls=() port
  for port in "$@"; do urls+=("http://127.0.0.1:$port"); done
  log "eval $name / $mode on ${#urls[@]} server(s)"
  # Lines are prefixed with the arm, since both arms share session.log.
  # shellcheck disable=SC2086
  "$PYTHON" "$EVAL" run --base-url "${urls[@]}" --concurrency "$CONCURRENCY" \
    $args --output "$SESSION_DIR/runs/$name/$mode" \
    > >(awk -v p="[$mode] " '{ print p $0; fflush() }') 2>&1 &
  EVAL_PIDS+=("$!")
}

stop_evals() {
  (( ${#EVAL_PIDS[@]} )) || return 0
  local pid
  for pid in "${EVAL_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${EVAL_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  EVAL_PIDS=()
}

wait_evals() {  # $1 = task name, $2 = deadline (epoch s, 0 = none); 1 if cut off
  local name="$1" deadline="$2" last pid
  last=$(date +%s)
  while :; do
    local running=0
    for pid in "${EVAL_PIDS[@]}"; do alive "$pid" && running=1; done
    (( running )) || break
    if (( deadline > 0 && $(date +%s) >= deadline )); then
      log "time budget reached: stopping evals"
      stop_evals
      return 1
    fi
    if (( $(date +%s) - last >= REPORT_EVERY_MIN * 60 )); then
      report "$name"; pack; last=$(date +%s)
    fi
    sleep 15
  done
  local failed=0
  for pid in "${EVAL_PIDS[@]}"; do wait "$pid" || failed=1; done
  EVAL_PIDS=()
  (( failed )) && log "WARNING: an eval of $name exited non-zero"
  return 0
}

on_exit() {
  local code=$?
  stop_evals
  stop_servers
  for task in "${TASKS[@]}"; do report "${task%%|*}"; done
  pack || true
  log "exit code $code; session dir $SESSION_DIR"
  log "pull it from the laptop: bash benchmark/looped_transformer/fetch_results.sh <you>@<node>"
}
trap on_exit EXIT
trap 'exit 130' INT TERM
# ssh disconnect outside tmux: the terminal (and the tee into session.log)
# goes away, so log straight to the file, then clean up like Ctrl-C.
# Without this, bash dies on SIGHUP and the setsid'd servers keep the GPUs.
trap 'exec >> "$SESSION_DIR/session.log" 2>&1; exit 129' HUP

# --------------------------------------------------------------------------
# Session metadata (what was run, on which commit and machine)
# --------------------------------------------------------------------------
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo "git_head: $(git rev-parse HEAD)"
  echo "git_branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "git_status:"; git status --porcelain
  echo "model: $MODEL"
  echo "loop_config: $LOOP_CONFIG"
  echo "SGLANG_ENABLE_JIT_DEEPGEMM=$SGLANG_ENABLE_JIT_DEEPGEMM"
  echo "GPUS=$GPUS (parallel=$PARALLEL; baseline ${BASE_GPUS[*]}; loop ${LOOP_GPUS[*]})"
  echo "CONCURRENCY=$CONCURRENCY per server; TIME_BUDGET_MIN=$TIME_BUDGET_MIN"
  echo "ATTENTION_BACKEND=$ATTENTION_BACKEND (GPU compute capability ${GPU_CAP:-n/a})"
  echo "HF_HOME=${HF_HOME:-<unset>}"
  echo "tasks:"; printf '  %s\n' "${TASKS[@]}"
  echo "python: $("$PYTHON" --version 2>&1)"
} > "$SESSION_DIR/session_info.txt"
"$PYTHON" - > "$SESSION_DIR/packages.txt" <<'EOF'
from importlib.metadata import distributions
print("\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata["Name"]}, key=str.lower)))
EOF
nvidia-smi > "$SESSION_DIR/nvidia-smi.txt" 2>&1 || true
if (( ${#GPU_LIST[@]} >= 3 && ${#GPU_LIST[@]} % 2 )); then
  log "WARNING: odd GPU count; GPU ${GPU_LIST[${#GPU_LIST[@]}-1]} stays idle"
fi
if (( PARALLEL )); then
  log "session $SESSION_DIR, ${#TASKS[@]} task(s); baseline on GPUs ${BASE_GPUS[*]}, loop on GPUs ${LOOP_GPUS[*]}, in parallel"
else
  log "session $SESSION_DIR, ${#TASKS[@]} task(s); 1 GPU (${BASE_GPUS[0]}): baseline, then loop"
fi
log "attention backend: $ATTENTION_BACKEND (GPU compute capability ${GPU_CAP:-n/a})"

# --------------------------------------------------------------------------
# Preflight: fail in seconds, not after an hour
# --------------------------------------------------------------------------
"$PYTHON" -c "import sglang, datasets" \
  || { log "ERROR: sglang/datasets not importable in this venv"; exit 1; }
for task in "${TASKS[@]}"; do
  name="${task%%|*}"; args="${task#*|}"
  log "preflight dataset: $name"
  # shellcheck disable=SC2086
  "$PYTHON" "$EVAL" run --prepare-only $args \
    || { log "ERROR: dataset for $name failed to load (gated GPQA needs a local --data-file)"; exit 1; }
done

# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
start_group() {  # $1 = mode, rest = GPUs; starts one server per GPU
  local mode="$1" gpu port; shift
  GROUP_PORTS=()
  for gpu in "$@"; do
    port=$(( PORT + ${#SERVER_PIDS[@]} ))
    start_server "$mode" "$gpu" "$port" "$SESSION_DIR/servers/${CURRENT_TASK}_${mode}_gpu${gpu}.log"
    GROUP_PORTS+=("$port")
  done
}

failed=0
for task in "${TASKS[@]}"; do
  name="${task%%|*}"; args="${task#*|}"
  CURRENT_TASK="$name"
  if (( DEADLINE > 0 && $(date +%s) >= DEADLINE )); then
    log "time budget used up; skipping $name"; break
  fi
  t=$(date +%s)
  if (( PARALLEL )); then
    log "=== $name: baseline + loop in parallel ==="
    start_group baseline "${BASE_GPUS[@]}"; base_ports=("${GROUP_PORTS[@]}")
    start_group loop "${LOOP_GPUS[@]}"; loop_ports=("${GROUP_PORTS[@]}")
    wait_servers || { failed=1; break; }
    start_arm baseline "$name" "$args" "${base_ports[@]}"
    start_arm loop "$name" "$args" "${loop_ports[@]}"
    cut=0; wait_evals "$name" "$DEADLINE" || cut=1
    stop_servers
  else
    cut=0
    for mode in baseline loop; do
      log "=== $name / $mode ==="
      start_group "$mode" "${BASE_GPUS[0]}"
      wait_servers || { failed=1; break 2; }
      deadline="$DEADLINE"
      if [[ "$mode" == baseline && "$DEADLINE" -gt 0 ]]; then
        # Leave the loop arm as much time as the baseline gets.
        deadline=$(( $(date +%s) + (DEADLINE - $(date +%s)) / 2 ))
      fi
      start_arm "$mode" "$name" "$args" "${GROUP_PORTS[@]}"
      wait_evals "$name" "$deadline" || cut=1
      stop_servers
      pack
    done
  fi
  log "$name took $(( ($(date +%s) - t) / 60 )) min$( (( cut )) && echo ' (cut off by the time budget)')"
  report "$name"
  pack
done

log "all tasks done (failed=$failed)"
[[ -f "$SESSION_DIR/SUMMARY.md" ]] && cat "$SESSION_DIR/SUMMARY.md"
exit "$failed"
