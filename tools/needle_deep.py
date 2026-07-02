#!/usr/bin/env python3
# Deep needle gate for q27 fp8 KV: ~355K-token haystack (War and Peace),
# needles at 10/35/60/70/78/95% depth (last two beyond the 262K native limit).
# Run deepest-first so each later (shallower) request hits the prefix cache.
import json
import time
import urllib.request

BODY = open("/tmp/claude-1000/-home-gabe/31483d5a-e57d-4270-9a5e-2b2fb32c8bbc/scratchpad/wandp_body.txt").read()
HAY = BODY[:1455000]  # ~355K tokens at ~4.10 chars/tok

NEEDLES = [  # (depth, needle, question, expected substring)
    (0.95, "The calibration constant for the tidal array is 88231.",
     "What is the calibration constant for the tidal array?", "88231"),
    (0.78, "The secret passphrase for the blue vault is 'cobalt-fern-42'.",
     "What is the secret passphrase for the blue vault?", "cobalt-fern-42"),
    (0.70, "The maintenance code for the orbital elevator is 917403.",
     "What is the maintenance code for the orbital elevator?", "917403"),
    (0.60, "Dr. Imura's favorite lichen species is Xanthoria parietina.",
     "What is Dr. Imura's favorite lichen species?", "Xanthoria parietina"),
    (0.35, "The registration number of the airship Peregrine is ZK-4471.",
     "What is the registration number of the airship Peregrine?", "ZK-4471"),
    (0.10, "The harbor master's cat is named Bramblewick.",
     "What is the name of the harbor master's cat?", "Bramblewick"),
]

def build(depth, needle):
    at = int(len(HAY) * depth)
    nl = HAY.rfind("\n", 0, at)
    if nl < 0: nl = at
    return HAY[:nl] + "\n" + needle + "\n" + HAY[nl:]

ok_native, n_native, ok_beyond, n_beyond = 0, 0, 0, 0
for depth, needle, q, want in NEEDLES:
    doc = build(depth, needle)
    body = {"model": "q27",
            "messages": [{"role": "user",
                          "content": doc + "\n\nBased on the document above: " + q +
                                     " Answer in one short sentence."}],
            "max_tokens": 800}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        resp = json.load(r)
    dt = time.time() - t0
    text = resp["choices"][0]["message"]["content"]
    ptok = resp.get("usage", {}).get("prompt_tokens", -1)
    tok_depth = int(depth * 355000 / 1000)
    hit = want.lower() in text.lower()
    beyond = depth * 355000 > 262144
    if beyond: n_beyond += 1; ok_beyond += hit
    else: n_native += 1; ok_native += hit
    tag = "BEYOND-NATIVE" if beyond else "native"
    print(f"depth {depth:.0%} (~{tok_depth}K tok, {tag}): "
          f"{'PASS' if hit else 'FAIL'} in {dt:.0f}s (prompt {ptok})", flush=True)
    print(f"  tail: {text[-140:]!r}", flush=True)
print(f"\nwithin-native: {ok_native}/{n_native}  beyond-native: {ok_beyond}/{n_beyond}")
