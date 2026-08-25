#!/usr/bin/env python3
"""Fold a Q27_DRIFT_CORPUS capture into the committed drift corpus.

    tools/corpus_dedup.py [--out tools/drift_corpus] capture.jsonl [more.jsonl ...]
    make corpus-dedup CORPUS=/path/to/capture.jsonl

Input is the JSONL src/drift_capture.h writes: one record per dialect-bearing
turn, already redacted. Records are grouped by `id` (the shape hash), one
exemplar is kept per shape with a `count`, and two things are written:

    <out>/corpus.jsonl   one exemplar per shape, descending count, no `ts`
    <out>/seeds/<id>     the exemplar's redacted text, for `make fuzz`

An existing <out>/corpus.jsonl is merged, so counts accumulate across capture
sessions and a hand-written `shape` label survives the next fold. `ts` is
dropped on purpose: a record's time of day is the one field that says
something about the session rather than the model, and the corpus is public.

Prints the shape histogram. Read it before touching the parser: it is the
first look at what the model actually emits, weighted by how often.
"""
import argparse
import json
import os
import sys

REQUIRED = ("id", "redacted", "outcome")


def load_records(path):
    """Yield (record, line_number) for each parseable line; warn on the rest."""
    bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(rec, dict) or any(k not in rec for k in REQUIRED):
                bad += 1
                continue
            yield rec, n
    if bad:
        print(f"warning: {path}: skipped {bad} unreadable line(s)", file=sys.stderr)


def fold(existing, records):
    """Merge records into the shape table. Returns (table, new_shape_ids)."""
    table = {r["id"]: r for r in existing}
    new_ids = set()
    for rec, _ in records:
        sid = rec["id"]
        outcome = rec.get("outcome", "")
        if sid in table:
            ex = table[sid]
            ex["count"] = ex.get("count", 1) + 1
            ex.setdefault("outcomes", {})
            ex["outcomes"][outcome] = ex["outcomes"].get(outcome, 0) + 1
            ex["bytes_max"] = max(ex.get("bytes_max", ex.get("bytes", 0)), rec.get("bytes", 0))
            continue
        new_ids.add(sid)
        table[sid] = {
            "id": sid,
            "shape": rec.get("shape", "") or "",
            "tags": rec.get("tags", []),
            "outcome": outcome,
            "outcomes": {outcome: 1},
            "redacted": rec["redacted"],
            "bytes": rec.get("bytes", 0),
            "bytes_max": rec.get("bytes", 0),
            "count": 1,
        }
    return table, new_ids


def dominant_outcome(rec):
    outcomes = rec.get("outcomes") or {rec.get("outcome", ""): rec.get("count", 1)}
    return max(outcomes.items(), key=lambda kv: (kv[1], kv[0]))[0]


def one_line(text, width=64):
    text = text.replace("\r", "").replace("\n", "\\n")
    return text if len(text) <= width else text[: width - 3] + "..."


def write_corpus(out_dir, rows):
    os.makedirs(os.path.join(out_dir, "seeds"), exist_ok=True)
    corpus_path = os.path.join(out_dir, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for rec in rows:
            rec = dict(rec)
            rec.pop("ts", None)
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    for rec in rows:
        with open(os.path.join(out_dir, "seeds", rec["id"]), "w", encoding="utf-8") as f:
            f.write(rec["redacted"])
    return corpus_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+", help="JSONL file(s) written by Q27_DRIFT_CORPUS")
    ap.add_argument("--out", default="tools/drift_corpus", help="corpus directory (default: tools/drift_corpus)")
    ap.add_argument("--dry-run", action="store_true", help="print the histogram, write nothing")
    args = ap.parse_args(argv)

    existing = []
    corpus_path = os.path.join(args.out, "corpus.jsonl")
    if os.path.exists(corpus_path):
        existing = [r for r, _ in load_records(corpus_path)]

    total = 0
    records = []
    for path in args.captures:
        for item in load_records(path):
            records.append(item)
            total += 1

    table, new_ids = fold(existing, records)
    rows = sorted(table.values(), key=lambda r: (-r.get("count", 1), r["id"]))

    print(f"{'count':>6}  {'outcome':<20} {'tags':<26} {'id':<16}  redacted")
    for rec in rows:
        tags = ",".join(rec.get("tags", []))
        mark = "*" if rec["id"] in new_ids else " "
        label = rec.get("shape") or one_line(rec["redacted"])
        print(f"{rec.get('count', 1):>6}{mark} {dominant_outcome(rec):<20} {tags:<26} {rec['id']:<16}  {label}")
    print(
        f"\n{total} record(s) read from {len(args.captures)} file(s); "
        f"{len(rows)} shape(s), {len(new_ids)} new; {sum(r.get('count', 1) for r in rows)} total across sessions"
        + ("  (* = new this fold)" if new_ids else "")
    )

    if args.dry_run:
        return 0
    written = write_corpus(args.out, rows)
    print(f"wrote {written} and {len(rows)} seed(s) under {os.path.join(args.out, 'seeds')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
