#!/usr/bin/env python3
"""End-to-end CLI fixture for Qwen3.8 standalone base + MTP GGUF input."""

import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np
from gguf import GGUFWriter


ROOT = Path(__file__).resolve().parent.parent
REPACK = ROOT / "tools" / "repack.py"
SPEC = importlib.util.spec_from_file_location("q27_repack_e2e", REPACK)
repack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repack)
MAGIC = 0x46373251


def write_gguf(path, block_count, tensors, *, nextn=None):
    writer = GGUFWriter(path, "qwen35")
    writer.add_name("Qwen3.8-27B")
    writer.add_block_count(block_count)
    if nextn is not None:
        writer.add_uint32("qwen35.nextn_predict_layers", nextn)
    for name, value in repack.QWEN35_REQUIRED_METADATA.items():
        if isinstance(value, list):
            writer.add_array(name, value)
        elif isinstance(value, float):
            writer.add_float32(name, value)
        else:
            writer.add_uint32(name, value)
    for name, tensor in tensors:
        writer.add_tensor(name, tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def read_q27(path):
    with path.open("rb") as stream:
        magic, version, count, meta_size = struct.unpack("<IIII", stream.read(16))
        assert magic == MAGIC and version == 1
        metadata = json.loads(stream.read(meta_size))
        entries = {}
        for _ in range(count):
            name_size, = struct.unpack("<H", stream.read(2))
            name = stream.read(name_size).decode()
            dtype, rank = struct.unpack("<BB", stream.read(2))
            shape = tuple(struct.unpack("<Q", stream.read(8))[0]
                          for _ in range(rank))
            offsets = struct.unpack("<QQQQ", stream.read(32))
            assert name not in entries
            entries[name] = {"dtype": dtype, "shape": shape, "offsets": offsets}
    return metadata, entries


def main():
    with tempfile.TemporaryDirectory(prefix="q27-repack-split-e2e.") as raw:
        work = Path(raw)
        base = work / "Qwen3.8-27B-BF16.gguf"
        mtp = work / "mtp-Qwen3.8-27B-BF16.gguf"
        output = work / "candidate.q27"
        launcher = work / "fixture_repack_cli.py"
        # Keep the serialized fixture tiny while exercising repack.main's real
        # argparse/GGUF/output path. Helper-level tests exercise the immutable
        # production dimensions; this wrapper scales only fixture dimensions.
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
            return shared_values.get(
                name, np.linspace(-3, 3, 128, dtype=np.float32))

        shared = [(name, shared_values[name]) for name in shared_values]
        base_only = sorted(repack.QWEN35_BASE_TENSORS - set(shared_values))
        mtp_only = sorted(repack.QWEN35_MTP_TENSORS)
        write_gguf(base, 64, shared + [
            (name, fixture_tensor(name)) for name in base_only
        ])
        write_gguf(mtp, 65, shared + [
            (name, fixture_tensor(name)) for name in mtp_only
        ], nextn=1)

        rejected = subprocess.run(
            [sys.executable, str(launcher), str(base), str(output), "--q4-head"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        assert rejected.returncode != 0, rejected.stdout
        assert "base-only qwen35 GGUF" in rejected.stderr, rejected.stderr
        assert not output.exists()

        subprocess.run([
            sys.executable, str(launcher), str(base), str(output),
            "--mtp", str(mtp), "--q4-head", "--tag", "q38-split-e2e-v1",
            "--report", "0",
        ], check=True)
        metadata, entries = read_q27(output)
        assert metadata["general.architecture"] == "qwen35"
        assert metadata["general.name"] == "Qwen3.8-27B"
        assert metadata["qwen35.block_count"] == 65
        assert metadata["qwen35.nextn_predict_layers"] == 1
        assert metadata["quant_policy"] == "q38-split-e2e-v1"
        assert len(entries) == 866
        assert set(entries) == (repack.QWEN35_BASE_TENSORS |
                                repack.QWEN35_MTP_TENSORS)
        assert entries["output.weight"]["dtype"] == 3  # Q4_G64 (--q4-head)
        assert entries["blk.64.attn_q.weight"]["dtype"] == 2  # Q8_G128
        assert entries["output_norm.weight"]["dtype"] == 0  # F32

    print("split MTP GGUF end-to-end repack: PASS")


if __name__ == "__main__":
    main()
