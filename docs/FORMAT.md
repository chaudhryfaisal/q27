# q27 weight format (version 1)

Offline-repacked weights for the q27 engine. Produced by `tools/repack.py` from the BF16 GGUF. Designed for mmap + single cudaMemcpy per tensor, and coalesced 128-byte warp reads in the fused-dequant GEMV.

## Container layout

```
[header]
  magic      u32   = 0x46373251  ("Q27F" little-endian)
  version    u32   = 1
  n_tensors  u32
  meta_len   u32
  meta       u8[meta_len]   JSON: arch config + layer map + quant policy
[tensor table]  n_tensors entries:
  name_len   u16
  name       u8[name_len]   GGUF tensor name, unchanged
  dtype      u8             0=F32  1=F16  2=Q8_G128  3=Q4_G64  4=T2_G128  5=T3_G128  6=B1_G128
  n_dims     u8
  shape      u64[n_dims]    numpy row-major shape (outer first; innermost/contiguous LAST)
  data_off   u64            relative to data section start, 256-byte aligned
  data_size  u64
  scale_off  u64            0 if dtype has no scales
  scale_size u64
[data section]  256-byte aligned blobs
```

## Quantized types

All quantized types quantize along the **contiguous (innermost) axis**, which
is the GEMV reduction axis for every matmul weight in this model.

### Q4_G64 (bulk weights)
- symmetric, group size 64: `scale = max(|wـgroup|) / 7`, `q = clip(round(w/scale), -8, 7) + 8` stored as unsigned nibble
- packing: element `i` of a row -> byte `i/2`; **even index = low nibble**, odd = high nibble
- scales: fp16, shape `[rows, cols/64]`, separate contiguous blob
- effective 4.25 bpw
- a warp reading 128 B gets 256 consecutive weights = exactly 4 groups

### Q8_G128 (quality-sensitive weights)
- symmetric, group size 128: `scale = max(|w_group|) / 127`, int8
- scales: fp16, `[rows, cols/128]`
- effective 8.125 bpw

### T2_G128 (ternary weights, dtype 4)
- group size 128; element `i` of a row uses the 2-bit field at
  `(i % 4) * 2` in byte `i / 4` (sequential, LSB-first)
- code `c` must be in `{0,1,2}` and decodes to `(c-1) * scale`; code 3 is invalid
- row data is `cols/4` bytes; scales are fp16 `[rows, cols/128]`
- effective 2.125 bits per weight

### B1_G128 (binary weights, dtype 6)
- group size 128; element `i` of a row uses bit `i % 8` in byte `i / 8`
  (sequential, LSB-first)
- bit `b` decodes to `(2b-1) * scale`; every bit pattern is valid
- row data is `cols/8` bytes; scales are fp16 `[rows, cols/128]`
- effective 1.125 bits per weight

### T3_G128 (experimental ternary packing, dtype 5)
- group size 128; five ternary codes are stored per byte in base 3:
  `c0 + 3*c1 + 9*c2 + 27*c3 + 81*c4`, with every byte in `[0,242]`
- each group uses 26 bytes; byte 25 carries columns 125..127 and its unused
  slots 3 and 4 must contain code 1 (zero)
- row data is `(cols/128)*26` bytes; scales are fp16 `[rows, cols/128]`
- effective 1.75 bits per weight; dtype 5 is reserved and no production
  artifacts currently use it

## Quant policy (v1)

| tensors | dtype | why |
|---|---|---|
| all `*_norm.weight`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, `output_norm` | F32 | tiny, numerically sensitive |
| `ssm_alpha.weight`, `ssm_beta.weight` | F16 | 48-wide heads, awkward group size, tiny anyway |
| `token_embd.weight`, `output.weight` | Q8_G128 | vocab quality; embed is row-lookup (not GEMV-read) |
| everything in `blk.64.*` (MTP layer) | Q8_G128 (matmuls) / F32 (norms) | draft/verify agreement must survive quantization or MTP acceptance craters |
| all other matmul weights (blk.0-63) | Q4_G64 | the ~14 GB bulk |

This table is the DEFAULT tier (~5.25 bpw overall). The q6 / q6k quality tiers
(2026-07-12, BUILDLOG) promote selected bulk tensors to Q8_G128 within the same
container and dtype set; the tier is recorded in the header meta as
`quant_policy` (e.g. `q6-v1`). No new dtypes, no version bump.

## Per-step read budget (decode)

Q4 bulk ~13.2 GB + Q8 lm_head ~1.3 GB + MTP layer ~0.4 GB + f16/f32 small tensors
=> ~14.8-15 GB per verify step. 5090 @ 1.79 TB/s => ~120 t/s ceiling before MTP amortization.
