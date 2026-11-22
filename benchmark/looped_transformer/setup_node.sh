#!/usr/bin/env bash
# Fresh GPU node -> ready for run_session.sh. No sudo, no credentials.
#
#   git clone -b shuwang/looped https://github.com/shuwang21/sglang.git ~/sglang-looped
#   bash ~/sglang-looped/benchmark/looped_transformer/setup_node.sh
#
# Safe to rerun: finished steps are skipped or cheap. It does:
#   1. uv and a minimal Rust toolchain (sglang's editable install builds Rust)
#   2. the checkout at origin/$BRANCH, if it is not there yet
#   3. a uv venv (managed Python 3.12), the hash-locked deps, sglang editable
#   4. CUDA from pip wheels (the node has no system CUDA):
#      - nvcc/crt/nvvm pinned to 13.0.x to match the 13.0 runtime and
#        headers (the lock's 13.5 nvcc fails flashinfer's CCCL check)
#      - lib64 -> lib and libcudart.so symlinks for JIT links
#      - JIT caches cleared so nothing built with the wrong nvcc is reused
#   5. ~/env.sh (PATH, HF_HOME, CUDA_HOME, venv); source it in every shell
#   6. checks, then the model and MMLU-Pro downloads (anonymous)
#
# Env: REPO_URL, BRANCH (shuwang/looped), DIR (~/sglang-looped),
#      HF_HOME (~/hf_cache), MODEL (openai/gpt-oss-20b), SKIP_DOWNLOAD=1.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/shuwang21/sglang.git}"
BRANCH="${BRANCH:-shuwang/looped}"
DIR="${DIR:-$HOME/sglang-looped}"
HF_HOME="${HF_HOME:-$HOME/hf_cache}"
MODEL="${MODEL:-openai/gpt-oss-20b}"
LOCK=benchmark/looped_transformer/requirements-linux.lock.txt
T0=$(date +%s)

step() { printf '\n== [+%ds] %s\n' $(( $(date +%s) - T0 )) "$*"; }
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

step "1. uv and Rust"
# Check our own install paths, not just "uv on PATH": some nodes ship a snap
# uv (/snap/bin/uv) that refuses to run with a home outside /home
# ("home directories outside of /home needs configuration").
[[ -x "$HOME/.local/bin/uv" ]] || curl -LsSf https://astral.sh/uv/install.sh | sh
[[ -x "$HOME/.cargo/bin/cargo" ]] \
  || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
hash -r
for tool in uv cargo; do
  case "$(command -v "$tool")" in
    /snap/*) echo "ERROR: $tool resolves to $(command -v "$tool") (snap)"; exit 1 ;;
  esac
done
uv --version
cargo --version

step "2. checkout $BRANCH in $DIR"
if [[ ! -d "$DIR/.git" ]]; then
  git clone -b "$BRANCH" "$REPO_URL" "$DIR"
fi
cd "$DIR"
git log --oneline -1

step "3. venv, locked dependencies, sglang (editable)"
[[ -x .venv/bin/python ]] || uv venv --seed --python 3.12 --python-preference only-managed .venv
# shellcheck disable=SC1091
. .venv/bin/activate
uv pip install --require-hashes -r "$LOCK"
uv pip install --no-cache --no-deps --no-build-isolation -e python

step "4. CUDA from pip wheels"
uv pip install "nvidia-cuda-nvcc~=13.0.0" "nvidia-cuda-crt~=13.0.0" "nvidia-nvvm~=13.0.0"
CUDA_HOME="$(python -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')"
[[ -e "$CUDA_HOME/lib64" ]] || ln -s lib "$CUDA_HOME/lib64"
if [[ ! -e "$CUDA_HOME/lib/libcudart.so" ]]; then
  cudart="$(cd "$CUDA_HOME/lib" && ls libcudart.so.13* | head -n 1)"
  ln -s "$cudart" "$CUDA_HOME/lib/libcudart.so"
fi
rm -rf "$HOME/.cache/sglang/jit" "$HOME/.cache/flashinfer"
echo "CUDA_HOME=$CUDA_HOME"

step "5. ~/env.sh"
cat > "$HOME/env.sh" <<EOF
# written by setup_node.sh; source in every new shell: . ~/env.sh
export PATH=\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH
export HF_HOME=$HF_HOME
export SGLANG_ENABLE_JIT_DEEPGEMM=0
cd $DIR && . .venv/bin/activate
export PYTHONPATH=$DIR/python
export CUDA_HOME=$CUDA_HOME
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
EOF
set +u
# shellcheck disable=SC1091
. "$HOME/env.sh"
set -u

step "6. checks"
nvcc --version | tail -n 2
python - <<'EOF'
import torch, sglang
n = torch.cuda.device_count()
print(f"torch {torch.__version__}, cuda available {torch.cuda.is_available()}, {n} GPU(s)")
print(f"sglang from {sglang.__file__}")
assert n > 0, "no GPU visible"
EOF
nvcc --version | grep -q "release 13.0" \
  || { echo "ERROR: nvcc is not 13.0; flashinfer JIT will fail"; exit 1; }

if [[ "${SKIP_DOWNLOAD:-0}" != 1 ]]; then
  step "7. downloads (no login needed)"
  mkdir -p "$HF_HOME"
  hf download "$MODEL" > /dev/null
  python -c "from datasets import load_dataset as l; print(l('TIGER-Lab/MMLU-Pro', split='test'))"
fi

step "done in $(( ($(date +%s) - T0) / 60 )) min"
cat <<EOF
Next, inside tmux:
  tmux new -s gptoss
  . ~/env.sh && TIME_BUDGET_MIN=<minutes left minus 25> bash benchmark/looped_transformer/run_session.sh
and on the laptop, pull results every 10 minutes:
  bash benchmark/looped_transformer/fetch_results.sh --watch $(whoami)@<node>
EOF
