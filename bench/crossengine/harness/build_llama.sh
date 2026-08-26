#!/usr/bin/env bash
# Update llama.cpp to current master and rebuild for BOTH archs on this box.
#
# -DCMAKE_CUDA_ARCHITECTURES="86;120": a 5090-only build silently fails kernels
# on the 3090, and this checkout is shared with dual-GPU work. sm_120 is
# consumer Blackwell -- NOT sm_100/101, which is the server part.
set -euo pipefail
D=/mnt/ai/projects/llama.cpp
cd "$D"

echo "=== before: $(git rev-parse --short HEAD) ==="
git stash list | head -3
git checkout master
git pull --ff-only origin master
echo "=== after:  $(git rev-parse --short HEAD) $(git log -1 --format=%cd --date=short) ==="

export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;120" -DLLAMA_CURL=OFF
cmake --build build --config Release -j "$(nproc)" --target llama-server llama-cli

echo "=== built ==="
ls -la build/bin/llama-server
./build/bin/llama-server --version 2>&1 | head -3
