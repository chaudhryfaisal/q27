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
    ("token_embd.weight", 2, [1, 128], 128, 2),
    ("blk.0.ffn_gate.weight", 5, [1, 128], 26, 2),
    ("blk.3.attn_q.weight", 6, [1, 128], 16, 2),
    ("blk.64.nextn.eh_proj.weight", 4, [1, 128], 32, 2),
    ("output_norm.weight", 0, [4], 16, 0),
    ("test.f16", 1, [2], 4, 0),
    ("test.q4", 3, [1, 64], 32, 2),
]


def align(value):
    return (value + ALIGN - 1) // ALIGN * ALIGN


def make_fixture(path, corrupt_size=False, misaligned=False, bad_t2=False,
                 bad_t3_range=False, bad_t3_padding=False, overflow=False,
                 bad_offset=False, zero_scale_offset=False, bad_dtype=False,
                 bad_data_alignment=False, bad_scale_alignment=False,
                 nonzero_absent_scale_offset=False):
    meta = b'{}'
    entries = []
    blobs = bytearray()

    for index, (name, dtype, shape, data_size, scale_size) in enumerate(TENSORS):
        if bad_dtype and index == 1:
            dtype, data_size, scale_size = 255, 0, 0
        if misaligned and index == 0:
            shape = [1, 129]
        if overflow and dtype == 5:
            shape = [(1 << 63) + 1, 128]
        data_off = align(len(blobs))
        stored_data_off = (1 << 64) - 256 if bad_offset and dtype == 4 else data_off
        if bad_data_alignment and index == 0:
            stored_data_off += 1
        blobs.extend(b'\0' * (data_off - len(blobs)))
        stored_data_size = data_size + (1 if corrupt_size and index == 1 else 0)
        data = bytearray(bytes([index + 1]) * stored_data_size)
        if dtype == 5:
            # The final T3 byte carries three real codes followed by two
            # canonical zero padding codes (1*27 + 1*81).
            data[25] = 108
            if bad_t3_range:
                data[0] = 243
            if bad_t3_padding:
                data[25] = 0
        if dtype == 4 and bad_t2:
            data[0] = 3
        blobs.extend(data)

        scale_off = 0
        if scale_size:
            scale_off = align(len(blobs))
            blobs.extend(b'\0' * (scale_off - len(blobs)))
            blobs.extend(b'\0<' * (scale_size // 2))
            if zero_scale_offset and index == 1:
                scale_off = 0
        stored_scale_off = scale_off
        if bad_scale_alignment and index == 0:
            stored_scale_off += 1
        if nonzero_absent_scale_offset and index == 4:
            stored_scale_off = ALIGN

        name_bytes = name.encode()
        entry = bytearray(struct.pack('<H', len(name_bytes)))
        entry.extend(name_bytes)
        entry.extend(struct.pack('<BB', dtype, len(shape)))
        entry.extend(struct.pack('<' + 'Q' * len(shape), *shape))
        entry.extend(struct.pack('<QQQQ', stored_data_off, stored_data_size,
                                 stored_scale_off, scale_size))
        entries.append(entry)

    header = struct.pack('<IIII', MAGIC, VERSION, len(entries), len(meta)) + meta
    table = b''.join(entries)
    data_base = align(len(header) + len(table))
    path.write_bytes(header + table + b'\0' * (data_base - len(header) - len(table)) + blobs)


def run(inspect, fixture):
    return subprocess.run([inspect, str(fixture)], text=True, capture_output=True)


def output(result):
    return result.stdout + result.stderr


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f'usage: {sys.argv[0]} build/inspect')
    inspect = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix='q27-inspect-') as tmp:
        root = pathlib.Path(tmp)
        valid = root / 'valid.q27'
        corrupt = root / 'corrupt.q27'
        misaligned = root / 'misaligned.q27'
        bad_t2 = root / 'bad-t2.q27'
        bad_t3_range = root / 'bad-t3-range.q27'
        bad_t3_padding = root / 'bad-t3-padding.q27'
        overflow = root / 'overflow.q27'
        bad_offset = root / 'bad-offset.q27'
        bad_dtype = root / 'bad-dtype.q27'
        zero_scale_offset = root / 'zero-scale-offset.q27'
        bad_data_alignment = root / 'bad-data-alignment.q27'
        bad_scale_alignment = root / 'bad-scale-alignment.q27'
        nonzero_absent_scale_offset = root / 'nonzero-absent-scale-offset.q27'
        make_fixture(valid)
        make_fixture(corrupt, corrupt_size=True)
        make_fixture(misaligned, misaligned=True)
        make_fixture(bad_t2, bad_t2=True)
        make_fixture(bad_t3_range, bad_t3_range=True)
        make_fixture(bad_t3_padding, bad_t3_padding=True)
        make_fixture(overflow, overflow=True)
        make_fixture(bad_dtype, bad_dtype=True)

        make_fixture(bad_offset, bad_offset=True)
        make_fixture(zero_scale_offset, zero_scale_offset=True)
        make_fixture(bad_data_alignment, bad_data_alignment=True)
        make_fixture(bad_scale_alignment, bad_scale_alignment=True)
        make_fixture(nonzero_absent_scale_offset,
                     nonzero_absent_scale_offset=True)
        good = run(inspect, valid)
        if good.returncode != 0 or '\nOK\n' not in good.stdout:
            sys.stderr.write(good.stdout + good.stderr)
            raise SystemExit('valid packed-dtype fixture failed inspection')

        invalid_dtype = run(inspect, bad_dtype)
        if (invalid_dtype.returncode == 0 or
                'unsupported dtype 255' not in output(invalid_dtype)):
            sys.stderr.write(output(invalid_dtype))
            raise SystemExit('undeclared packed dtype was not rejected')


        bad = run(inspect, corrupt)
        if (bad.returncode == 0 or
                'invalid tensor payload blk.0.ffn_gate.weight' not in output(bad)):
            sys.stderr.write(output(bad))
            raise SystemExit('one-byte packed-dtype corruption was not detected')

        bad_shape = run(inspect, misaligned)
        if (bad_shape.returncode == 0 or
                'not divisible by group 128' not in output(bad_shape)):
            sys.stderr.write(output(bad_shape))
            raise SystemExit('misaligned packed-dtype shape was not detected')

        invalid_t2 = run(inspect, bad_t2)
        if (invalid_t2.returncode == 0 or
                'T2 payload contains reserved code 3' not in output(invalid_t2)):
            sys.stderr.write(output(invalid_t2))
            raise SystemExit('reserved T2 code was not detected')

        invalid_t3_range = run(inspect, bad_t3_range)
        if (invalid_t3_range.returncode == 0 or
                'T3 payload byte exceeds 242' not in output(invalid_t3_range)):
            sys.stderr.write(output(invalid_t3_range))
            raise SystemExit('out-of-range T3 byte was not detected')

        invalid_t3_padding = run(inspect, bad_t3_padding)
        if (invalid_t3_padding.returncode == 0 or
                'T3 final-byte padding is noncanonical' not in output(invalid_t3_padding)):
            sys.stderr.write(output(invalid_t3_padding))
            raise SystemExit('noncanonical T3 padding was not detected')

        invalid_overflow = run(inspect, overflow)
        if (invalid_overflow.returncode == 0 or
                'tensor element count overflows uint64' not in
                invalid_overflow.stdout + invalid_overflow.stderr):
            sys.stderr.write(invalid_overflow.stdout + invalid_overflow.stderr)
            raise SystemExit('packed payload size overflow was not detected')

        invalid_offset = run(inspect, bad_offset)
        if (invalid_offset.returncode == 0 or
                'tensor data out of range' not in invalid_offset.stdout + invalid_offset.stderr):
            sys.stderr.write(invalid_offset.stdout + invalid_offset.stderr)
            raise SystemExit('overflowing tensor offset was not detected')
        invalid_scale_offset = run(inspect, zero_scale_offset)
        if (invalid_scale_offset.returncode == 0 or
                'tensor scale offset is zero' not in output(invalid_scale_offset)):
            sys.stderr.write(output(invalid_scale_offset))
            raise SystemExit('zero scale offset was not detected')
        invalid_data_alignment = run(inspect, bad_data_alignment)
        if (invalid_data_alignment.returncode == 0 or
                'tensor data offset is not 256-byte aligned' not in
                output(invalid_data_alignment)):
            sys.stderr.write(output(invalid_data_alignment))
            raise SystemExit('unaligned tensor data offset was not detected')
        invalid_scale_alignment = run(inspect, bad_scale_alignment)
        if (invalid_scale_alignment.returncode == 0 or
                'tensor scale offset is not 256-byte aligned' not in
                output(invalid_scale_alignment)):
            sys.stderr.write(output(invalid_scale_alignment))
            raise SystemExit('unaligned tensor scale offset was not detected')
        invalid_absent_scale = run(inspect, nonzero_absent_scale_offset)
        if (invalid_absent_scale.returncode == 0 or
                'tensor scale offset must be zero without scales' not in
                output(invalid_absent_scale)):
            sys.stderr.write(output(invalid_absent_scale))
            raise SystemExit('nonzero absent-scale offset was not detected')

    print('inspect packed dtype fixtures: PASS')


if __name__ == '__main__':
    main()
