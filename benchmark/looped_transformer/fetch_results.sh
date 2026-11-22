#!/usr/bin/env bash
# Run on your LAPTOP (not the GPU node): copy session results from a node over
# ssh, then commit each session to the results branch with push_results.sh.
# The shared GPU nodes must not hold GitHub / HF credentials, so results are
# pulled from here instead of pushed from there.
#
#   bash benchmark/looped_transformer/fetch_results.sh [--watch[=SEC]] <ssh-host> [remote_repo_dir]
#
# <ssh-host>        what you pass to ssh, e.g. user@node or a ~/.ssh/config alias
# remote_repo_dir   the checkout on the node, relative to its home
#                   (default: sglang-looped)
# --watch[=SEC]     keep fetching every SEC seconds (default 600) until Ctrl-C.
#                   Allocations are reclaimed without warning and the disk is
#                   wiped, so leave this running while a session runs.
#
# Safe to run repeatedly while a session is still going (rsync is incremental;
# an unchanged session is not re-committed). Local copies go to
# ../looped-sessions/ next to this checkout (override with DEST), outside
# the code worktree.

set -euo pipefail

watch=0
case "${1:-}" in
  --watch) watch=600; shift ;;
  --watch=*) watch="${1#--watch=}"; shift ;;
esac
host="${1:?usage: fetch_results.sh [--watch[=SEC]] <ssh-host> [remote_repo_dir]}"
rdir="${2:-sglang-looped}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
dest="${DEST:-$(dirname "$repo")/looped-sessions}"
mkdir -p "$dest"

fetch_once() {
  echo "fetch_results [$(date +%H:%M:%S)]: $host:$rdir/results/gptoss_sessions/ -> $dest/"
  rsync -a --exclude '*.tar.gz' --exclude '*.tmp' \
    "$host:$rdir/results/gptoss_sessions/" "$dest/" || return 1
  shopt -s nullglob
  local sessions=("$dest"/session_*/) session
  if (( ${#sessions[@]} == 0 )); then
    echo "fetch_results: no session_* directories found"
    return 1
  fi
  for session in "${sessions[@]}"; do
    bash "$here/push_results.sh" "${session%/}" \
      || echo "fetch_results: WARNING: push of $session failed (local copy kept)"
  done
  for session in "${sessions[@]}"; do
    [[ -f "$session/SUMMARY.md" ]] && echo "summary: ${session}SUMMARY.md"
  done
  return 0
}

if (( watch == 0 )); then
  fetch_once
  exit $?
fi
while :; do
  fetch_once || echo "fetch_results: WARNING: fetch failed; retrying in ${watch}s"
  sleep "$watch"
done
