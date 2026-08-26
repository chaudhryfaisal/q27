#!/usr/bin/env python3
"""Re-splice src/server.cu's build_prompt / prepare_anthropic_prompt / handle
lambdas into tools/test_chat_completions_integration.cpp, byte-for-byte.

The inverse of tools/extract_check.sh. Run it after any edit to handle() in
server.cu, then fix whatever the harness scaffolding needs (a new captured
variable, a new lambda parameter) and re-run the harness. The harness had
drifted for ten server.cu commits before make test-tools started running
extract_check.sh first; this makes catching up a one-liner.

    python3 tools/extract_sync.py && ./tools/extract_check.sh
"""
import os
import sys

MARKERS = ('auto build_prompt = [&]', 'auto prepare_anthropic_prompt = [&]', 'auto handle = [&]')


def extract(text, marker):
    i = text.index(marker)
    j = text.index('{', i)
    depth = 0
    k = j
    while True:
        c = text[k]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i, k + 1
        k += 1


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real = open(os.path.join(root, 'src/server.cu')).read()
    harness_path = os.path.join(root, 'tools/test_chat_completions_integration.cpp')
    fake = open(harness_path).read()
    changed = 0
    for marker in MARKERS:
        ri, rk = extract(real, marker)
        fi, fk = extract(fake, marker)
        if fake[fi:fk] != real[ri:rk]:
            fake = fake[:fi] + real[ri:rk] + fake[fk:]
            changed += 1
    open(harness_path, 'w').write(fake)
    print(f"{changed} lambda(s) re-spliced" if changed else "already in sync")
    return 0


if __name__ == '__main__':
    sys.exit(main())
