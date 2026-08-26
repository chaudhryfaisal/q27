#!/usr/bin/env python3
"""Prove tools/repack_canonical_gate.sh actually gates.

A gate nobody has seen fail is not a gate. This drives the real script over a
spec-shrunk split fixture (the same trick tools/test_repack_split_e2e.py uses)
and asserts all four branches: SKIP with no config, SKIP on a missing source,
PASS on the true digest, and MISMATCH on a wrong one.
"""

import hashlib
import os
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

ROOT = Path(__file__).resolve().parent.parent
REPACK = ROOT / "tools" / "repack.py"
GATE = ROOT / "tools" / "repack_canonical_gate.sh"

SPEC = importlib.util.spec_from_file_location("q27_repack_gate_test", REPACK)
repack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repack)


def write_gguf(path, block_count, tensors, *, nextn=None):
    w = GGUFWriter(path, "qwen35")
    w.add_name("Qwen3.8-27B")
    w.add_block_count(block_count)
    if nextn is not None:
        w.add_uint32("qwen35.nextn_predict_layers", nextn)
    for name, value in repack.QWEN35_REQUIRED_METADATA.items():
        if isinstance(value, list):
            w.add_array(name, value)
        elif isinstance(value, float):
            w.add_float32(name, value)
        else:
            w.add_uint32(name, value)
    for name, tensor in tensors:
        w.add_tensor(name, tensor)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def build_fixture(work):
    """Tiny split pair + a launcher that shrinks only the fixture dimensions."""
    base, mtp = work / "base.gguf", work / "mtp.gguf"
    launcher = work / "fixture_repack_cli.py"
    launcher.write_text(f'''\
import importlib.util
spec = importlib.util.spec_from_file_location("q27_repack_fixture", {str(REPACK)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.QWEN35_BASE_SPECS = {{name: ("F32", (128,)) for name in module.QWEN35_BASE_TENSORS}}
module.QWEN35_MTP_SPECS = {{name: ("F32", (128,)) for name in module.QWEN35_MTP_TENSORS}}
module.main()
''', encoding="utf-8")

    shared_values = {
        "token_embd.weight": np.linspace(-1, 1, 128, dtype=np.float32),
        "output_norm.weight": np.linspace(0, 1, 128, dtype=np.float32),
        "output.weight": np.linspace(-2, 2, 128, dtype=np.float32),
    }

    def fixture_tensor(name):
        return shared_values.get(name, np.linspace(-3, 3, 128, dtype=np.float32))

    shared = [(n, shared_values[n]) for n in shared_values]
    base_only = sorted(repack.QWEN35_BASE_TENSORS - set(shared_values))
    mtp_only = sorted(repack.QWEN35_MTP_TENSORS)
    write_gguf(base, 64, shared + [(n, fixture_tensor(n)) for n in base_only])
    write_gguf(mtp, 65, shared + [(n, fixture_tensor(n)) for n in mtp_only], nextn=1)
    return base, mtp, launcher


def run_gate(env_extra, cwd):
    # Inherit the real environment: the gate shells out to a Python that needs
    # the interpreter's own site-packages. Only the gate's own knobs are
    # overridden, and the ones not set for a given case are cleared so a
    # leftover value cannot make a SKIP case look configured.
    env = dict(os.environ)
    for knob in ("SRC_GGUF", "SRC_MTP_GGUF", "CANON_MD5",
                 "REPACK_CMD", "REPACK_ARGS"):
        env.pop(knob, None)
    env.update({k: str(v) for k, v in env_extra.items()})
    return subprocess.run(["bash", str(GATE)], env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False)


def main():
    with tempfile.TemporaryDirectory(prefix="q27-gate-test.") as raw:
        work = Path(raw)
        base, mtp, launcher = build_fixture(work)
        cmd = f"{sys.executable} {launcher}"
        args = "--q4-head --tag q38-split-gate-test-v1 --report 0"

        # unconfigured -> SKIP, exit 0 (so it can sit in a pipeline always)
        r = run_gate({}, work)
        assert r.returncode == 0 and "SKIP" in r.stdout, r.stdout

        # configured but source absent -> SKIP, exit 0
        r = run_gate({"SRC_GGUF": work / "nope.gguf", "CANON_MD5": "x"}, work)
        assert r.returncode == 0 and "SKIP" in r.stdout, r.stdout

        # derive the true digest of a split repack, then assert PASS on it
        out = work / "truth.q27"
        subprocess.run([sys.executable, str(launcher), str(base), str(out),
                        "--mtp", str(mtp), "--q4-head",
                        "--tag", "q38-split-gate-test-v1", "--report", "0"],
                       check=True, stdout=subprocess.DEVNULL)
        truth = hashlib.md5(out.read_bytes()).hexdigest()

        r = run_gate({"SRC_GGUF": base, "SRC_MTP_GGUF": mtp, "CANON_MD5": truth,
                      "REPACK_CMD": cmd, "REPACK_ARGS": args}, work)
        assert r.returncode == 0, r.stdout
        assert "PASS (split)" in r.stdout, r.stdout

        # wrong digest -> MISMATCH, non-zero. This is the branch that matters:
        # every shape/manifest gate still passes, only the bytes disagree.
        r = run_gate({"SRC_GGUF": base, "SRC_MTP_GGUF": mtp,
                      "CANON_MD5": "0" * 32,
                      "REPACK_CMD": cmd, "REPACK_ARGS": args}, work)
        assert r.returncode != 0, r.stdout
        assert "MD5 MISMATCH" in r.stdout, r.stdout

    print("repack canonical gate: PASS")


if __name__ == "__main__":
    main()
