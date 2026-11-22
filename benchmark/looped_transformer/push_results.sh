#!/usr/bin/env bash
# Commit a result directory to a dedicated results branch and push it.
#
#   bash benchmark/looped_transformer/push_results.sh <dir> [name]
#
# <dir> is stored on the branch as <name>/ (default: basename of <dir>);
# re-running replaces that folder with the current contents. Nothing is
# checked out: the commit is built with a temporary index, so the current
# branch, working tree and index are left untouched. Locally, read it with
#   git fetch origin shuwang/looped-results
#
# Env: RESULTS_BRANCH (default shuwang/looped-results),
#      RESULTS_REMOTE (default origin).

set -euo pipefail

src="${1:?usage: push_results.sh <dir> [name]}"
[[ -d "$src" ]] || { echo "push_results: no such directory: $src" >&2; exit 1; }
src="$(cd "$src" && pwd)"
name="${2:-$(basename "$src")}"
branch="${RESULTS_BRANCH:-shuwang/looped-results}"
remote="${RESULTS_REMOTE:-origin}"

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

# Never block on a credential prompt: this runs unattended inside
# run_session.sh. Without cached credentials the push fails and the caller
# only logs a warning. (HTTPS: GIT_TERMINAL_PROMPT; SSH: BatchMode.)
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes}"

# Commits need an identity; fall back to a neutral one on fresh nodes.
if [[ -z "$(git config user.email || true)" ]]; then
  export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-looped-session}"
  export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-looped-session@localhost}"
  export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/stage"
cp -R "$src" "$tmp/stage/$name"
export GIT_INDEX_FILE="$tmp/index"

for attempt in 1 2 3; do
  parent=""
  if git fetch -q "$remote" "refs/heads/$branch" 2>/dev/null; then
    parent="$(git rev-parse FETCH_HEAD)"
    git read-tree "$parent"
  else
    git read-tree --empty  # first push creates the branch (no code history)
  fi
  git rm -r -q --cached --ignore-unmatch -- "$name" > /dev/null
  git --work-tree="$tmp/stage" add -A -f -- "$name"
  tree="$(git write-tree)"
  if [[ -n "$parent" && "$tree" == "$(git rev-parse "$parent^{tree}")" ]]; then
    echo "push_results: $name unchanged on $branch"
    exit 0
  fi
  commit="$(git commit-tree "$tree" ${parent:+-p "$parent"} \
    -m "results: $name ($(date -u +%Y-%m-%dT%H:%M:%SZ), $(hostname), code $(git rev-parse --short HEAD))")"
  if git push -q "$remote" "$commit:refs/heads/$branch"; then
    echo "push_results: pushed $name -> $remote/$branch ($(git rev-parse --short "$commit"))"
    exit 0
  fi
  echo "push_results: push rejected (attempt $attempt), refetching" >&2
  sleep 2
done
echo "push_results: FAILED to push $name to $remote/$branch" >&2
exit 1
