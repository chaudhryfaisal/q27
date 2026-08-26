#!/usr/bin/env python3
"""Label the drift corpus: what the model INTENDED each shape to be.

    tools/corpus_label.py propose [--all]        derive intended from the structure, mark "proposed"
    tools/corpus_label.py list [proposed|human|unlabelled]
    tools/corpus_label.py show ID
    tools/corpus_label.py confirm ID [ID ...]    a human agrees with the proposal
    tools/corpus_label.py set ID '<json calls>' [--proposed]   write the label (human unless --proposed)
    tools/corpus_label.py clear ID
    tools/corpus_label.py unreliable ID [ID ...] [--note TEXT]  replay of this row is not evidence
                                                (captured by an older redactor); leaves the denominator

`intended` is the oracle and is human-decided; a proposal is a second,
independent reading of the redacted structure (regex, no parser) so the
parser's reading can be diffed against it by `make corpus-check`. Where the
two agree the label is probably right; where they disagree a human looks.
Confirming turns "proposed" into "human". Labels live in the corpus rows and
survive `make corpus-dedup` (existing rows are kept on refold).

Placeholder values (PATH_1, CODE_2:ml) are the label's values: that is what
the parser sees in the redacted text, so the comparison is exact.
"""
import json
import re
import sys

CORPUS = "tools/drift_corpus/corpus.jsonl"
OPENER = re.compile(r'<function=([A-Za-z0-9_.:-]+)>|<tool_name>\s*([A-Za-z0-9_.:-]+)\s*</tool_name>|<parameter=([A-Z][A-Za-z0-9_]*)>')
PARAM = re.compile(r'<parameter=([a-z_][A-Za-z0-9_]*)>\n?(.*?)(?=\n?</parameter>|\n?<parameter=|\n?</function>|\n?<tool_call>|\n?</tool_call>|\Z)', re.S)
JSON_CALL = re.compile(r'"name"\s*:\s*"([A-Za-z0-9_.:-]+)"')
JSON_KV = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*(?:"([^"]*)"|([A-Za-z0-9_.:]+))')


def load():
    with open(CORPUS, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save(rows):
    with open(CORPUS, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def xml_calls(text):
    """Calls read off the XML dialect: an opener, then its parameters."""
    calls = []
    for m in OPENER.finditer(text):
        name = m.group(1) or m.group(2) or m.group(3)
        start = m.end()
        nxt = OPENER.search(text, start)
        end = nxt.start() if nxt else len(text)
        args = {}
        for pm in PARAM.finditer(text, start, end):
            args[pm.group(1)] = pm.group(2).strip()
        calls.append({"name": name, "arguments": args})
    return calls


def json_calls(text):
    """Calls read off the JSON dialect, leniently: each "name" then its keys."""
    calls = []
    names = list(JSON_CALL.finditer(text))
    for i, m in enumerate(names):
        end = names[i + 1].start() if i + 1 < len(names) else len(text)
        args = {}
        for kv in JSON_KV.finditer(text, m.end(), end):
            k = kv.group(1)
            if k in ("name", "arguments", "tool_call"):
                continue
            args[k] = kv.group(2) if kv.group(2) is not None else kv.group(3)
        calls.append({"name": m.group(1), "arguments": args})
    return calls


def propose(row):
    text = row["redacted"]
    calls = xml_calls(text) if ("<function=" in text or "<parameter=" in text or "<tool_name>" in text) else json_calls(text)
    # an undeclared name (NAME_n) is a call the model wanted but nothing can execute
    calls = [c for c in calls if not c["name"].startswith("NAME_")]
    return {"calls": calls, "by": "proposed"}


def one_line(text, width=70):
    text = text.replace("\n", "\\n")
    return text if len(text) <= width else text[: width - 3] + "..."


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    rows = load()
    by_id = {r["id"]: r for r in rows}
    if cmd == "propose":
        n = 0
        for r in rows:
            if "intended" in r and not (args and args[0] == "--all"):
                continue
            if r.get("intended", {}).get("by") == "human":
                continue
            r["intended"] = propose(r)
            n += 1
        save(rows)
        print(f"proposed {n} label(s)")
    elif cmd == "list":
        want = args[0] if args else None
        for r in rows:
            lab = r.get("intended")
            state = "unlabelled" if not lab else lab.get("by", "?")
            if want and state != want:
                continue
            calls = ", ".join(f"{c['name']}({','.join(c.get('arguments', {}))})" for c in (lab or {}).get("calls", []))
            print(f"{r['id']}  {state:<10} {r.get('outcome',''):<18} x{r.get('count',1):<4} {calls or '-':<50} {one_line(r['redacted'], 60)}")
    elif cmd == "show":
        r = by_id[args[0]]
        print(json.dumps({k: r[k] for k in ("id", "outcome", "tags", "count") if k in r}, indent=1))
        print(r["redacted"])
        print("intended:", json.dumps(r.get("intended"), indent=1))
        print("current: ", json.dumps(r.get("current"), indent=1))
    elif cmd == "confirm":
        for sid in args:
            r = by_id[sid]
            if "intended" not in r:
                r["intended"] = propose(r)
            r["intended"]["by"] = "human"
        save(rows)
        print(f"confirmed {len(args)}")
    elif cmd == "set":
        r = by_id[args[0]]
        r["intended"] = {"calls": json.loads(args[1]), "by": "proposed" if "--proposed" in args else "human"}
        save(rows)
        print("set")
    elif cmd == "unreliable":
        note = None
        ids = []
        i = 0
        while i < len(args):
            if args[i] == "--note":
                note = args[i + 1]
                i += 2
            else:
                ids.append(args[i])
                i += 1
        for sid in ids:
            by_id[sid]["replay"] = "unreliable"
            if note:
                by_id[sid]["note"] = note
        save(rows)
        print(f"marked {len(ids)} unreliable")
    elif cmd == "clear":
        by_id[args[0]].pop("intended", None)
        save(rows)
        print("cleared")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
