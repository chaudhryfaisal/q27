#!/usr/bin/env bash
# Two open questions, both blocking the cross-engine table.
#
# 1. llama.cpp ladder OOMs at --parallel 8 on the GDN "rs cache" (recurrent
#    state), ~1047 MiB PER SLOT on top of 18.19 GiB of weights.
#    --ctx-checkpoints was the wrong knob; that sizes context checkpoints, not
#    recurrent state. Find the largest slot count that actually boots.
#
# 2. vLLM emits CORRUPTED generations after ~3 agentic instances -- incoherent
#    token soup returned as HTTP 200, reproduced across two runs. Isolate
#    whether it is the MTP speculative path or the prefix cache, by running the
#    same 4 instances three ways.
set -u
OUT=/mnt/ai/projects/club-3090/results/ninfer-vs-q27-20260817
H=$OUT/harness
source "$H/legs.sh"
ISO=$OUT/isolation; mkdir -p "$ISO"

# ---------- 1. llama slot ceiling ----------
for PAR in 8 6 4 2; do
  echo "### llama --parallel $PAR"
  LOG=$ISO/llama.par$PAR.log
  LLAMA_PAR=$PAR nohup "$LLAMA_BIN" -m "$LLAMA_GGUF" --port 8199 --host 127.0.0.1 \
    --device CUDA0 --device-draft CUDA0 -ngl 999 \
    --ctx-size $((PAR*16384)) --parallel "$PAR" \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 6 --reasoning off \
    --jinja --alias llama >"$LOG" 2>&1 &
  PID=$!
  ok=0
  for i in $(seq 1 120); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8199/health 2>/dev/null)" = "200" ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || break
    sleep 2
  done
  if [ $ok = 1 ]; then
    echo "  BOOTED at --parallel $PAR (VRAM $(vram_now) MiB)"
    grep -aoE "rs cache[^,]*|KV self size[^,]*" "$LOG" | head -2
    kill $PID 2>/dev/null; sleep 8
    echo "$PAR" >"$ISO/llama_max_parallel.txt"
    break
  fi
  echo "  FAILED at --parallel $PAR: $(grep -aoE 'allocating [0-9.]+ MiB|failed to allocate buffer for rs cache' "$LOG" | head -2 | tr '\n' ' ')"
  kill -9 $PID 2>/dev/null; sleep 5
done

# ---------- 2. vLLM corruption isolation ----------
# 4 instances is enough: corruption appeared at instance 3-4 in both prior runs.
FOUR="pallets__flask-5014 psf__requests-1142 psf__requests-1724 psf__requests-1766"

run_vllm_variant() {
  local name="$1"; shift
  echo "### vllm variant: $name"
  docker rm -f ab-vllm >/dev/null 2>&1; sleep 5
  docker run -d --name ab-vllm --gpus '"device=0"' --ipc=host \
    -p 8198:8000 -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$VLLM_IMAGE" --model "$VLLM_MODEL" --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.93 --max-model-len 106496 --max-num-seqs 16 \
    --host 0.0.0.0 --served-model-name vllm claude-opus-4-8 \
    claude-haiku-4-5-20251001 claude-sonnet-4-5 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --trust-remote-code --kv-cache-dtype fp8 --quantization compressed-tensors \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    "$@" >"$ISO/vllm.$name.boot.log" 2>&1
  for i in $(seq 1 150); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8198/health 2>/dev/null)" = "200" ] && break
    [ "$(docker inspect --format='{{.State.Status}}' ab-vllm 2>/dev/null)" = "exited" ] && { echo "  DIED"; return 1; }
    sleep 5
  done
  echo "  booted"
  python3 "$H/tapproxy.py" --listen 8081 --upstream 127.0.0.1:8198 \
    --log "$ISO/tap.vllm.jsonl" --strip-fields output_config,thinking \
    --translate-thinking >/dev/null 2>&1 &
  local TP=$!
  sleep 2
  for IID in $FOUR; do
    SWEBENCH_WORK=/mnt/ai/swebench-work-iso bash "$H/agentic.sh" "vllm" 8081 "$ISO" "$IID" \
      >>"$ISO/agentic.vllm.$name.log" 2>&1
    mv "$ISO/agentic.vllm.jsonl" "$ISO/agentic.vllm.$name.jsonl.part" 2>/dev/null
    cat "$ISO/agentic.vllm.$name.jsonl.part" >>"$ISO/agentic.vllm.$name.jsonl" 2>/dev/null
    rm -f "$ISO/agentic.vllm.$name.jsonl.part"
  done
  kill $TP 2>/dev/null; sleep 2
  mv "$ISO/tap.vllm.jsonl" "$ISO/tap.vllm.$name.jsonl" 2>/dev/null
  docker logs ab-vllm >"$ISO/vllm.$name.container.log" 2>&1
  docker rm -f ab-vllm >/dev/null 2>&1; sleep 8
}

run_vllm_variant "specON_cacheON"  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
run_vllm_variant "specOFF_cacheON"
run_vllm_variant "specOFF_cacheOFF" --no-enable-prefix-caching

echo "### DONE"
