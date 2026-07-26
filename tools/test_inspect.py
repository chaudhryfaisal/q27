#!/usr/bin/env python3
import pathlib
import struct
import subprocess
import sys
import tempfile

MAGIC = 0x46373251
VERSION = 1
ALIGN = 256

TENSORS = [
    ("token_embd.weight", 4, [1, 128], 32, 2),
    ("blk.0.ffn_gate.weight", 5, [1, 128], 26, 2),
    ("blk.3.attn_q.weight", 6, [1, 128], 16, 2),
    ("blk.64.nextn.eh_proj.weight", 2, [1, 128], 128, 2),
    ("output_norm.weight", 0, [4], 16, 0),
    ("test.f16", 1, [2], 4, 0),
    ("test.q4", 3, [1, 64], 32, 2),
]


def align(value):
    return (value + ALIGN - 1) // ALIGN * ALIGN


def make_fixture(path, corrupt=False, misaligned=False):
    meta = b'{}'
    entries = []
    blobs = bytearray()

    for index, (name, dtype, shape, data_size, scale_size) in enumerate(TENSORS):
        if misaligned and index == 0:
            shape = [1, 129]
        data_off = align(len(blobs))
        blobs.extend(b'\0' * (data_off - len(blobs)))
        stored_data_size = data_size + (1 if corrupt and index == 1 else 0)
        data = bytearray(bytes([index + 1]) * stored_data_size)
        if dtype == 5:
            # The final T3 byte carries three real codes followed by two
            # canonical zero padding codes (1*27 + 1*81).
            data[25] = 108
        blobs.extend(data)

        scale_off = 0
        if scale_size:
            scale_off = align(len(blobs))
            blobs.extend(b'\0' * (scale_off - len(blobs)))
            blobs.extend(b'\0<' * (scale_size // 2))

        name_bytes = name.encode()
        entry = bytearray(struct.pack('<H', len(name_bytes)))
        entry.extend(name_bytes)
        entry.extend(struct.pack('<BB', dtype, len(shape)))
        entry.extend(struct.pack('<' + 'Q' * len(shape), *shape))
        entry.extend(struct.pack('<QQQQ', data_off, stored_data_size, scale_off, scale_size))
        entries.append(entry)

    header = struct.pack('<IIII', MAGIC, VERSION, len(entries), len(meta)) + meta
    table = b''.join(entries)
    data_base = align(len(header) + len(table))
    path.write_bytes(header + table + b'\0' * (data_base - len(header) - len(table)) + blobs)


def run(inspect, fixture):
    return subprocess.run([inspect, str(fixture)], text=True, capture_output=True)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f'usage: {sys.argv[0]} build/inspect')
    inspect = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix='q27-inspect-') as tmp:
        root = pathlib.Path(tmp)
        valid = root / 'valid.q27'
        corrupt = root / 'corrupt.q27'
        misaligned = root / 'misaligned.q27'
        make_fixture(valid)
        make_fixture(corrupt, corrupt=True)
        make_fixture(misaligned, misaligned=True)

        good = run(inspect, valid)
        if good.returncode != 0 or '\nOK\n' not in good.stdout:
            sys.stderr.write(good.stdout + good.stderr)
            raise SystemExit('valid packed-dtype fixture failed inspection')

        bad = run(inspect, corrupt)
        if bad.returncode == 0 or 'INVARIANT FAIL blk.0.ffn_gate.weight' not in bad.stdout:
            sys.stderr.write(bad.stdout + bad.stderr)
            raise SystemExit('one-byte packed-dtype corruption was not detected')

        bad_shape = run(inspect, misaligned)
        if (bad_shape.returncode == 0 or
                'not divisible by group 128' not in bad_shape.stdout):
            sys.stderr.write(bad_shape.stdout + bad_shape.stderr)
            raise SystemExit('misaligned packed-dtype shape was not detected')

    print('inspect packed dtype fixtures: PASS')


if __name__ == '__main__':
    main()
