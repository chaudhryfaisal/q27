#!/usr/bin/env python3
"""Convert the Qwopus3.6-27B-v2-MTP BF16 GGUF (arch qwen35) back to a HF
transformers-loadable safetensors checkpoint (Qwen3_5ForCausalLM, model_type
qwen3_5_text, MTP tensors preserved under top-level mtp.*).

This inverts llama.cpp's convert_hf_to_gguf.py mapping for Qwen3_5TextModel
(conversion/qwen.py: Qwen3NextModel.modify_tensors + _LinearAttentionVReorderBase
+ _Qwen35MtpMixin), verified against the shipped GGUF and the official
Qwen/Qwen3.6-27B model.safetensors.index.json tensor names.

Inverse transforms applied:
  1. all HF-side "*norm.weight" EXCEPT linear_attn.norm.weight: GGUF = HF + 1
     (zero-centered Qwen3_5RMSNorm) -> subtract 1.
  2. ssm_a = -exp(A_log) -> A_log = log(-ssm_a); plus V-head un-reorder.
  3. ssm_dt.bias -> linear_attn.dt_bias (rename); plus V-head un-reorder.
  4. ssm_conv1d [ch,4] -> conv1d [ch,1,4] (unsqueeze); V channel un-reorder.
  5. V-head reorder inversion (llama.cpp stores V heads tiled [G0v0,G1v0,...],
     HF stores grouped [G0v0,G0v1,G0v2,G1v0,...]) on: attn_qkv V rows,
     attn_gate rows, ssm_alpha rows, ssm_beta rows, ssm_a, ssm_dt.bias,
     ssm_conv1d V channels, ssm_out columns.
  6. attn_q (q/gate interleaved per head) is a direct copy: HF Qwen3_5Attention
     stores q_proj interleaved the same way (out = heads * head_dim * 2).

Usage:
  gguf_to_hf.py GGUF OUTDIR --scaffold SCAFFOLD_DIR [--shard-bytes N]
  gguf_to_hf.py GGUF OUTDIR --verify [N]
"""
import argparse
import json
import os
import random
import shutil
import sys

import numpy as np
import torch
from gguf import GGUFReader
from safetensors import safe_open
from safetensors.torch import save_file

# ---- architecture constants (asserted against GGUF metadata at load) --------
D_MODEL = 5120
N_LAYERS = 64          # main layers; layer 64 is the MTP block
MTP_LAYER = 64
N_FF = 17408
N_HEAD, N_KV, HEAD_DIM = 24, 4, 256
NKH = 16               # linear_num_key_heads
NVH = 48               # linear_num_value_heads
NVK = NVH // NKH       # v heads per k head = 3
HD_K = HD_V = 128      # linear key/value head dims
D_INNER = NVH * HD_V   # 6144
CONV_K = 4
QKV_ROWS_Q = NKH * HD_K            # 2048
QKV_ROWS_K = NKH * HD_K            # 2048
QKV_ROWS_V = NVH * HD_V            # 6144
CONV_CH = QKV_ROWS_Q + QKV_ROWS_K + QKV_ROWS_V  # 10240
VOCAB = 248320
GGUF_EOS = 248046

ATTN_LAYERS = set(range(3, N_LAYERS, 4)) | {MTP_LAYER}
SSM_LAYERS = set(range(N_LAYERS)) - set(range(3, N_LAYERS, 4))


# ---- V-head order permutations ----------------------------------------------
def unreorder_v(arr: np.ndarray, axis: int, head_dim: int) -> np.ndarray:
    """Inverse of llama.cpp _reorder_v_heads: tiled [NVK,NKH,hd] -> grouped [NKH,NVK,hd]."""
    shape = list(arr.shape)
    assert shape[axis] == NVK * NKH * head_dim, (shape, axis, head_dim)
    a = arr.reshape(shape[:axis] + [NVK, NKH, head_dim] + shape[axis + 1:])
    a = np.swapaxes(a, axis, axis + 1)
    return np.ascontiguousarray(a).reshape(shape)


def reorder_v(arr: np.ndarray, axis: int, head_dim: int) -> np.ndarray:
    """Forward llama.cpp _reorder_v_heads: grouped [NKH,NVK,hd] -> tiled [NVK,NKH,hd]."""
    shape = list(arr.shape)
    assert shape[axis] == NVK * NKH * head_dim, (shape, axis, head_dim)
    a = arr.reshape(shape[:axis] + [NKH, NVK, head_dim] + shape[axis + 1:])
    a = np.swapaxes(a, axis, axis + 1)
    return np.ascontiguousarray(a).reshape(shape)


# ---- GGUF tensor access ------------------------------------------------------
def gguf_np(t) -> np.ndarray:
    """GGUF tensor -> numpy array in row-major HF orientation.
    BF16 -> uint16 view (bit-exact); F32 -> float32."""
    tt = t.tensor_type.name
    raw = np.asarray(t.data)
    shape = tuple(reversed([int(d) for d in t.shape]))  # GGUF ne[0] is innermost
    if tt == "BF16":
        return raw.view(np.uint16).reshape(shape)
    if tt == "F32":
        return raw.view(np.float32).reshape(shape)
    raise AssertionError(f"{t.name}: unsupported GGUF dtype {tt} (need BF16/F32 source)")


def np_to_torch(arr: np.ndarray) -> torch.Tensor:
    if arr.dtype == np.uint16:
        return torch.from_numpy(np.ascontiguousarray(arr)).view(torch.bfloat16)
    assert arr.dtype == np.float32, arr.dtype
    return torch.from_numpy(np.ascontiguousarray(arr))


# ---- mapping -----------------------------------------------------------------
# transform tags:
#   copy      direct byte copy
#   minus1    f32, subtract 1 (zero-centered RMSNorm)
#   qkv_v     un-reorder V rows (rows 4096..10239, head_dim 128)
#   rows128   un-reorder all rows, head_dim 128
#   rows1     un-reorder all rows, head_dim 1
#   vec48     un-reorder 1D vector of 48
#   a_log     vec48 un-reorder then log(-x)
#   conv      un-reorder V channels then [ch,4] -> [ch,1,4]
#   cols128   un-reorder columns (axis 1), head_dim 128

GLOBAL_MAP = {
    "token_embd.weight": ("model.embed_tokens.weight", "copy"),
    "output.weight": ("lm_head.weight", "copy"),
    "output_norm.weight": ("model.norm.weight", "minus1"),
}

# leaf -> (hf_leaf, transform, valid_for)  valid_for: "ssm" | "attn" | "both"
BLK_MAP = {
    "attn_norm.weight": ("input_layernorm.weight", "minus1", "both"),
    "post_attention_norm.weight": ("post_attention_layernorm.weight", "minus1", "both"),
    "ffn_gate.weight": ("mlp.gate_proj.weight", "copy", "both"),
    "ffn_up.weight": ("mlp.up_proj.weight", "copy", "both"),
    "ffn_down.weight": ("mlp.down_proj.weight", "copy", "both"),
    # full attention (every 4th layer + MTP)
    "attn_q.weight": ("self_attn.q_proj.weight", "copy", "attn"),
    "attn_k.weight": ("self_attn.k_proj.weight", "copy", "attn"),
    "attn_v.weight": ("self_attn.v_proj.weight", "copy", "attn"),
    "attn_output.weight": ("self_attn.o_proj.weight", "copy", "attn"),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", "minus1", "attn"),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", "minus1", "attn"),
    # gated DeltaNet
    "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", "qkv_v", "ssm"),
    "attn_gate.weight": ("linear_attn.in_proj_z.weight", "rows128", "ssm"),
    "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", "rows1", "ssm"),
    "ssm_beta.weight": ("linear_attn.in_proj_b.weight", "rows1", "ssm"),
    "ssm_a": ("linear_attn.A_log", "a_log", "ssm"),
    "ssm_dt.bias": ("linear_attn.dt_bias", "vec48", "ssm"),
    "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "conv", "ssm"),
    "ssm_norm.weight": ("linear_attn.norm.weight", "copy", "ssm"),  # NOT zero-centered
    "ssm_out.weight": ("linear_attn.out_proj.weight", "cols128", "ssm"),
    # MTP extras (only on blk.64)
    "nextn.eh_proj.weight": ("__mtp__fc.weight", "copy", "attn"),
    "nextn.enorm.weight": ("__mtp__pre_fc_norm_embedding.weight", "minus1", "attn"),
    "nextn.hnorm.weight": ("__mtp__pre_fc_norm_hidden.weight", "minus1", "attn"),
    "nextn.shared_head_norm.weight": ("__mtp__norm.weight", "minus1", "attn"),
}

EXPECTED_SHAPES = {  # HF-orientation shapes, keyed by GGUF leaf (None = global)
    "token_embd.weight": (VOCAB, D_MODEL),
    "output.weight": (VOCAB, D_MODEL),
    "output_norm.weight": (D_MODEL,),
    "attn_norm.weight": (D_MODEL,),
    "post_attention_norm.weight": (D_MODEL,),
    "ffn_gate.weight": (N_FF, D_MODEL),
    "ffn_up.weight": (N_FF, D_MODEL),
    "ffn_down.weight": (D_MODEL, N_FF),
    "attn_q.weight": (N_HEAD * HEAD_DIM * 2, D_MODEL),
    "attn_k.weight": (N_KV * HEAD_DIM, D_MODEL),
    "attn_v.weight": (N_KV * HEAD_DIM, D_MODEL),
    "attn_output.weight": (D_MODEL, N_HEAD * HEAD_DIM),
    "attn_q_norm.weight": (HEAD_DIM,),
    "attn_k_norm.weight": (HEAD_DIM,),
    "attn_qkv.weight": (CONV_CH, D_MODEL),
    "attn_gate.weight": (D_INNER, D_MODEL),
    "ssm_alpha.weight": (NVH, D_MODEL),
    "ssm_beta.weight": (NVH, D_MODEL),
    "ssm_a": (NVH,),
    "ssm_dt.bias": (NVH,),
    "ssm_conv1d.weight": (CONV_CH, CONV_K),
    "ssm_norm.weight": (HD_V,),
    "ssm_out.weight": (D_MODEL, D_INNER),
    "nextn.eh_proj.weight": (D_MODEL, 2 * D_MODEL),
    "nextn.enorm.weight": (D_MODEL,),
    "nextn.hnorm.weight": (D_MODEL,),
    "nextn.shared_head_norm.weight": (D_MODEL,),
}


def map_name(gguf_name: str) -> tuple[str, str, str]:
    """GGUF tensor name -> (hf_name, transform, leaf). Asserts on anything unknown."""
    if gguf_name in GLOBAL_MAP:
        hf, tr = GLOBAL_MAP[gguf_name]
        return hf, tr, gguf_name
    parts = gguf_name.split(".", 2)
    assert len(parts) == 3 and parts[0] == "blk" and parts[1].isdecimal(), \
        f"unmapped GGUF tensor: {gguf_name}"
    bid, leaf = int(parts[1]), parts[2]
    assert leaf in BLK_MAP, f"unmapped GGUF block tensor: {gguf_name}"
    hf_leaf, tr, valid = BLK_MAP[leaf]
    is_attn = bid in ATTN_LAYERS
    assert valid == "both" or (valid == "attn") == is_attn, \
        f"{gguf_name}: leaf {leaf} not valid for {'attn' if is_attn else 'ssm'} layer {bid}"
    if hf_leaf.startswith("__mtp__"):
        assert bid == MTP_LAYER, f"{gguf_name}: nextn tensor outside MTP layer"
        return "mtp." + hf_leaf[len("__mtp__"):], tr, leaf
    if bid == MTP_LAYER:
        return f"mtp.layers.0.{hf_leaf}", tr, leaf
    assert 0 <= bid < N_LAYERS, f"{gguf_name}: bad layer id"
    return f"model.layers.{bid}.{hf_leaf}", tr, leaf


def apply_inverse(arr: np.ndarray, tr: str) -> np.ndarray:
    """GGUF layout/values -> HF layout/values."""
    if tr == "copy":
        return arr
    if tr == "minus1":
        assert arr.dtype == np.float32
        return arr - np.float32(1.0)
    if tr == "qkv_v":
        qk = arr[: QKV_ROWS_Q + QKV_ROWS_K]
        v = unreorder_v(arr[QKV_ROWS_Q + QKV_ROWS_K:], 0, HD_V)
        return np.ascontiguousarray(np.concatenate([qk, v], axis=0))
    if tr == "rows128":
        return unreorder_v(arr, 0, HD_V)
    if tr == "rows1":
        return unreorder_v(arr, 0, 1)
    if tr == "vec48":
        return unreorder_v(arr, 0, 1)
    if tr == "a_log":
        assert arr.dtype == np.float32
        a = unreorder_v(arr, 0, 1).astype(np.float64)
        assert np.all(a < 0), "ssm_a must be strictly negative (= -exp(A_log))"
        return np.log(-a).astype(np.float32)
    if tr == "conv":
        assert arr.shape == (CONV_CH, CONV_K) and arr.dtype == np.float32
        qk = arr[: QKV_ROWS_Q + QKV_ROWS_K]
        v = unreorder_v(arr[QKV_ROWS_Q + QKV_ROWS_K:], 0, HD_V)
        out = np.concatenate([qk, v], axis=0)
        return np.ascontiguousarray(out.reshape(CONV_CH, 1, CONV_K))
    if tr == "cols128":
        return unreorder_v(arr, 1, HD_V)
    raise AssertionError(f"unknown transform {tr}")


def apply_forward(arr: np.ndarray, tr: str) -> np.ndarray:
    """HF layout/values -> GGUF layout/values (for --verify)."""
    if tr == "copy":
        return arr
    if tr == "minus1":
        return arr + np.float32(1.0)
    if tr == "qkv_v":
        qk = arr[: QKV_ROWS_Q + QKV_ROWS_K]
        v = reorder_v(arr[QKV_ROWS_Q + QKV_ROWS_K:], 0, HD_V)
        return np.ascontiguousarray(np.concatenate([qk, v], axis=0))
    if tr == "rows128":
        return reorder_v(arr, 0, HD_V)
    if tr in ("rows1", "vec48"):
        return reorder_v(arr, 0, 1)
    if tr == "a_log":
        return reorder_v(-np.exp(arr.astype(np.float64)), 0, 1)  # stays f64; caller compares approx
    if tr == "conv":
        a = arr.reshape(CONV_CH, CONV_K)
        qk = a[: QKV_ROWS_Q + QKV_ROWS_K]
        v = reorder_v(np.ascontiguousarray(a[QKV_ROWS_Q + QKV_ROWS_K:]), 0, HD_V)
        return np.ascontiguousarray(np.concatenate([qk, v], axis=0))
    if tr == "cols128":
        return reorder_v(arr, 1, HD_V)
    raise AssertionError(f"unknown transform {tr}")


# ---- GGUF metadata sanity ----------------------------------------------------
def check_metadata(r: GGUFReader):
    def fv(key):
        f = r.fields[key]
        v = f.contents()
        return v.decode() if isinstance(v, bytes) else v

    assert fv("general.architecture") == "qwen35", fv("general.architecture")
    expect = {
        "qwen35.block_count": N_LAYERS + 1,
        "qwen35.embedding_length": D_MODEL,
        "qwen35.feed_forward_length": N_FF,
        "qwen35.attention.head_count": N_HEAD,
        "qwen35.attention.head_count_kv": N_KV,
        "qwen35.attention.key_length": HEAD_DIM,
        "qwen35.attention.value_length": HEAD_DIM,
        "qwen35.ssm.conv_kernel": CONV_K,
        "qwen35.ssm.state_size": HD_K,
        "qwen35.ssm.group_count": NKH,
        "qwen35.ssm.time_step_rank": NVH,
        "qwen35.ssm.inner_size": D_INNER,
        "qwen35.full_attention_interval": 4,
        "qwen35.nextn_predict_layers": 1,
        "tokenizer.ggml.eos_token_id": GGUF_EOS,
    }
    for k, want in expect.items():
        got = fv(k)
        assert got == want, f"GGUF metadata {k}: got {got}, expected {want}"


# ---- conversion --------------------------------------------------------------
def hf_sort_key(name: str):
    def lyr(n, prefix):
        return int(n[len(prefix):].split(".")[0])
    if name == "model.embed_tokens.weight":
        return (0, 0, name)
    if name.startswith("model.layers."):
        return (1, lyr(name, "model.layers."), name)
    if name == "model.norm.weight":
        return (2, 0, name)
    if name == "lm_head.weight":
        return (3, 0, name)
    if name.startswith("mtp."):
        return (4, 0, name)
    raise AssertionError(f"unexpected HF name {name}")


def convert(args):
    r = GGUFReader(args.gguf)
    check_metadata(r)
    tensors = {t.name: t for t in r.tensors}
    assert len(tensors) == len(r.tensors), "duplicate GGUF tensor names"

    # sanity: layer typing from actual tensor presence
    seen_attn = {int(n.split(".")[1]) for n in tensors if ".attn_q." in n}
    seen_ssm = {int(n.split(".")[1]) for n in tensors if ".ssm_out." in n}
    assert seen_attn == ATTN_LAYERS, f"attention layers mismatch: {sorted(seen_attn)}"
    assert seen_ssm == SSM_LAYERS, f"ssm layers mismatch: {sorted(seen_ssm)}"

    # build full mapping first; assert bijectivity and coverage
    mapping = {}  # gguf_name -> (hf_name, transform, leaf)
    for name in tensors:
        mapping[name] = map_name(name)
    hf_names = [m[0] for m in mapping.values()]
    assert len(set(hf_names)) == len(hf_names), "HF name collision"
    emb = tensors["token_embd.weight"]
    assert tuple(reversed([int(d) for d in emb.shape])) == (VOCAB, D_MODEL)

    # plan shards
    order = sorted(mapping.items(), key=lambda kv: hf_sort_key(kv[1][0]))
    sizes = {}
    for gname, (hfname, tr, leaf) in order:
        t = tensors[gname]
        n = 1
        for d in t.shape:
            n *= int(d)
        bpe = 2 if t.tensor_type.name == "BF16" else 4
        sizes[hfname] = n * bpe
    shards, cur, cur_bytes = [], [], 0
    for gname, (hfname, tr, leaf) in order:
        if cur and cur_bytes + sizes[hfname] > args.shard_bytes:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(gname)
        cur_bytes += sizes[hfname]
    if cur:
        shards.append(cur)
    n_shards = len(shards)
    total = sum(sizes.values())
    print(f"{len(order)} tensors, {total / 1e9:.2f} GB -> {n_shards} shards")

    os.makedirs(args.outdir, exist_ok=True)
    weight_map = {}
    for si, group in enumerate(shards, 1):
        fname = f"model-{si:05d}-of-{n_shards:05d}.safetensors"
        payload = {}
        for gname in group:
            hfname, tr, leaf = mapping[gname]
            t = tensors[gname]
            arr = gguf_np(t)
            exp = EXPECTED_SHAPES[leaf]
            assert arr.shape == exp, f"{gname}: shape {arr.shape} != expected {exp}"
            if tr != "copy":
                assert arr.dtype == np.float32 or tr in ("qkv_v", "rows128", "rows1", "cols128"), \
                    f"{gname}: transform {tr} on dtype {arr.dtype}"
            out = apply_inverse(arr, tr)
            payload[hfname] = np_to_torch(out)
            weight_map[hfname] = fname
        save_file(payload, os.path.join(args.outdir, fname), metadata={"format": "pt"})
        print(f"  wrote {fname} ({sum(p.numel() * p.element_size() for p in payload.values()) / 1e9:.2f} GB, {len(payload)} tensors)")
        del payload

    with open(os.path.join(args.outdir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2, sort_keys=True)

    write_configs(args, r)
    print("done.")


def write_configs(args, r: GGUFReader):
    scaf = args.scaffold
    base = json.load(open(os.path.join(scaf, "config.json")))
    assert base["architectures"] == ["Qwen3_5ForConditionalGeneration"], base["architectures"]
    text = base["text_config"]
    assert text["vocab_size"] == VOCAB
    assert text["tie_word_embeddings"] is False
    assert text["num_hidden_layers"] == N_LAYERS
    assert text["mtp_num_hidden_layers"] == 1
    assert text["full_attention_interval"] == 4
    # cross-check layer_types vs GGUF-derived layer map
    lt = text["layer_types"]
    assert {i for i, x in enumerate(lt) if x == "full_attention"} == ATTN_LAYERS - {MTP_LAYER}

    cfg = dict(text)
    cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    cfg["model_type"] = "qwen3_5_text"
    cfg["eos_token_id"] = GGUF_EOS  # <|im_end|>, per GGUF tokenizer.ggml.eos_token_id
    cfg["transformers_version"] = base.get("transformers_version", "4.57.1")
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    gen = json.load(open(os.path.join(scaf, "generation_config.json")))
    eos = gen.get("eos_token_id")
    assert GGUF_EOS in (eos if isinstance(eos, list) else [eos]), \
        f"generation_config eos {eos} missing GGUF eos {GGUF_EOS}"
    with open(os.path.join(args.outdir, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=2)

    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy(os.path.join(scaf, fn), os.path.join(args.outdir, fn))

    # chat template: prefer the one embedded in the GGUF (fine-tune ships a
    # modified template); fall back to the scaffold's.
    fld = r.fields.get("tokenizer.chat_template")
    if fld is not None:
        tpl = fld.contents()
        if isinstance(tpl, bytes):
            tpl = tpl.decode()
        with open(os.path.join(args.outdir, "chat_template.jinja"), "w") as f:
            f.write(tpl)
    else:
        shutil.copy(os.path.join(scaf, "chat_template.jinja"),
                    os.path.join(args.outdir, "chat_template.jinja"))


# ---- verify ------------------------------------------------------------------
def load_hf_tensor(outdir: str, weight_map: dict, hfname: str) -> np.ndarray:
    path = os.path.join(outdir, weight_map[hfname])
    with safe_open(path, framework="pt") as f:
        t = f.get_tensor(hfname)
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy()
    assert t.dtype == torch.float32, t.dtype
    return t.numpy()


def verify(args):
    r = GGUFReader(args.gguf)
    check_metadata(r)
    tensors = {t.name: t for t in r.tensors}
    weight_map = json.load(open(os.path.join(args.outdir, "model.safetensors.index.json")))["weight_map"]

    mapping = {name: map_name(name) for name in tensors}
    hf_set = {m[0] for m in mapping.values()}
    assert hf_set == set(weight_map), (
        f"index/weight_map key mismatch: missing={sorted(hf_set - set(weight_map))[:5]} "
        f"extra={sorted(set(weight_map) - hf_set)[:5]}")

    # sample: N random + at least one of every transform type
    rng = random.Random(args.seed)
    names = sorted(tensors)
    sample = set(rng.sample(names, min(args.verify, len(names))))
    by_tr = {}
    for name in names:
        by_tr.setdefault(mapping[name][1], []).append(name)
    for tr, group in sorted(by_tr.items()):
        if not (sample & set(group)):
            sample.add(rng.choice(group))

    failures = 0
    for gname in sorted(sample):
        hfname, tr, leaf = mapping[gname]
        g = gguf_np(tensors[gname])
        h = load_hf_tensor(args.outdir, weight_map, hfname)
        fwd = apply_forward(h, tr)
        if tr == "a_log":
            ref = g.astype(np.float64)
            rel = np.max(np.abs(fwd - ref) / np.maximum(np.abs(ref), 1e-30))
            ok = rel < 1e-6
            detail = f"max_rel={rel:.2e}"
        else:
            ok = fwd.shape == g.shape and fwd.dtype == g.dtype and \
                fwd.tobytes() == g.tobytes()
            detail = "byte-exact" if ok else "BYTES DIFFER"
        status = "ok " if ok else "FAIL"
        print(f"  [{status}] {gname:42s} -> {hfname:55s} {tr:8s} {detail}")
        if not ok:
            failures += 1
    print(f"verified {len(sample)} tensors, {failures} failures")
    if failures:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gguf")
    ap.add_argument("outdir")
    ap.add_argument("--scaffold", default=None,
                    help="dir with base Qwen/Qwen3.6-27B config.json + tokenizer files")
    ap.add_argument("--shard-bytes", type=int, default=5_000_000_000)
    ap.add_argument("--verify", type=int, nargs="?", const=24, default=None,
                    help="spot-check N random tensors against an existing OUTDIR")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.verify is not None:
        verify(args)
    else:
        assert args.scaffold, "--scaffold required for conversion"
        convert(args)


if __name__ == "__main__":
    main()
