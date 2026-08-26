# Drift Corpus Capture (Phase 1) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use conclave:executing-plans to implement this plan task-by-task.

**Goal:** Capture every dialect-bearing model turn as a redacted, deduplicated,
labelled corpus entry, so a later parser rewrite has an oracle built from real
traffic instead of invented cases.

**Architecture:** One new header `src/drift_capture.h` holding pure functions
(redaction, shape hashing, record formatting) that are unit-testable without
CUDA. `api_common.h`'s existing capture hook is widened from the
UN-RESCUED-only branch to every dialect-bearing turn and routed through it.
No parsing behaviour changes anywhere.

**Tech Stack:** C++17, nlohmann::json (vendored, `third_party/json.hpp`), the
repo's hand-rolled `CHECK()` test style (see `tools/test_openai_bridge.cpp`).

**Why this phase ships alone:** it is useful whether or not the rewrite
happens, and its output (current-vs-intended agreement) is what decides
whether the rewrite is worth doing at all. See
`docs/plans/2026-08-24-parser-normalize-then-parse-design.md`.

---

## Background the implementer needs

- **q27 is a public repo and the corpus is real Claude Code session content**:
  file paths, source code, tool arguments. Redaction is not optional and must
  happen *before* bytes reach disk. There is no scrubbing step later.
- **What the parser cares about is structure, not values.** Keep all framing
  (`<tool_call>`, `<function=NAME>`, `<parameter=KEY>`, JSON keys, tool names)
  verbatim. Replace only the *values*.
- **One exception that matters:** dialect markup appearing *inside* a value is
  structure, not content. A `</function>` inside a `content` value is the
  exact case that broke a previous fix. It must survive redaction.
- The existing hook is at `src/api_common.h:5573`, inside
  `else if (looks_like_intended_tool_call(text_in))` — i.e. **misses only**,
  raw, NUL-separated. A regression oracle needs the successes too.

---

### Task 1: Redaction core

**Files:**
- Create: `src/drift_capture.h`
- Test: `tools/test_drift_capture.cpp`

**Dependencies:** none

**Step 1: Write the failing test**

```cpp
// tools/test_drift_capture.cpp
#include "drift_capture.h"
#include <cstdio>
static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static void test_redacts_xml_values() {
    const std::string in =
        "<tool_call>\n<function=Read>\n<parameter=file_path>\n/home/gabe/secret.txt\n"
        "</parameter>\n</function>\n</tool_call>";
    const std::string out = q27::redact_drift(in);
    // framing and keys survive verbatim
    CHECK(out.find("<function=Read>") != std::string::npos);
    CHECK(out.find("<parameter=file_path>") != std::string::npos);
    // the value does not
    CHECK(out.find("/home/gabe/secret.txt") == std::string::npos);
    CHECK(out.find("PATH_1") != std::string::npos);
}

static void test_preserves_dialect_inside_values() {
    // a closer inside a value is STRUCTURE -- it must survive redaction,
    // because it is the shape that broke the closer-bounding fix
    const std::string in =
        "<function=Write>\n<parameter=content>\nfoo </function> bar\n</parameter>\n</function>";
    const std::string out = q27::redact_drift(in);
    CHECK(out.find("</function> ") != std::string::npos);
}

static void test_redacts_json_string_values() {
    const std::string in = R"({"name":"Read","arguments":{"file_path":"/etc/shadow"}})";
    const std::string out = q27::redact_drift(in);
    CHECK(out.find("\"name\":\"Read\"") != std::string::npos);   // tool name survives
    CHECK(out.find("\"file_path\"") != std::string::npos);        // key survives
    CHECK(out.find("/etc/shadow") == std::string::npos);          // value does not
}

static void test_placeholder_type_from_key() {
    const std::string in =
        "<function=Bash>\n<parameter=command>\nrm -rf /\n</parameter>\n"
        "<parameter=description>\nwipe\n</parameter>\n</function>";
    const std::string out = q27::redact_drift(in);
    CHECK(out.find("CODE_1") != std::string::npos);   // command -> CODE
    CHECK(out.find("TEXT_2") != std::string::npos);   // description -> TEXT
}

static void test_length_class_recorded() {
    std::string big = "<function=Write>\n<parameter=content>\n" + std::string(5000, 'x') +
                      "\n</parameter>\n</function>";
    const std::string out = q27::redact_drift(big);
    CHECK(out.find("TEXT_1:big") != std::string::npos);  // >4KB marked
    CHECK(out.size() < 500);                             // and not carried
}

int main() {
    test_redacts_xml_values();
    test_preserves_dialect_inside_values();
    test_redacts_json_string_values();
    test_placeholder_type_from_key();
    test_length_class_recorded();
    if (fails) { printf("%d FAILURE(S)\n", fails); return 1; }
    printf("DRIFT CAPTURE: all pass\n");
    return 0;
}
```

**Step 2: Run it to verify it fails**

```bash
g++ -std=c++17 -I src tools/test_drift_capture.cpp -o build/test_drift_capture
```
Expected: FAIL — `drift_capture.h: No such file or directory`.

**Step 3: Write the minimal implementation**

Create `src/drift_capture.h`. `redact_drift()` walks the text once and, for
each value region, emits `TYPE_N` (plus `:big` when >4096 bytes) while copying
any dialect markup found inside the value through verbatim.

Value regions are: between `<parameter=KEY>` and the matching `</parameter>`
(or the next `<parameter=` / `</function>` if the closer is missing — drift
means closers are often absent), and JSON string values whose key is not
`name`. Type from the key: `file_path|path` → `PATH`, `command|code|content`
→ `CODE`, anything else → `TEXT`. Index is 1-based per entry, in order of
appearance.

Dialect markup to pass through when found inside a value: `</function>`,
`</tool_call>`, `</parameter>`, `<function=`, `<parameter=`, `<tool_call>`.

**Step 4: Run it to verify it passes**

```bash
g++ -std=c++17 -I src tools/test_drift_capture.cpp -o build/test_drift_capture && ./build/test_drift_capture
```
Expected: `DRIFT CAPTURE: all pass`

**Step 5: Commit**

```bash
git add src/drift_capture.h tools/test_drift_capture.cpp
git commit -m "drift capture: redact values, keep framing (and dialect inside values)"
```

---

### Task 2: The leak gate

**Files:**
- Modify: `tools/test_drift_capture.cpp`

**Dependencies:** Task 1

This is the test that makes the public-repo claim true. It asserts no
identifiable byte run from any value survives redaction — the property the
whole phase rests on.

**Step 1: Write the failing test**

```cpp
static void test_no_value_bytes_survive() {
    // every distinctive value token must be absent from the output
    const char* secrets[] = {"sk-ant-api03-XXXX", "/home/gabe/.ssh/id_ed25519",
                             "hunter2", "AKIAIOSFODNN7EXAMPLE"};
    for (const char* s : secrets) {
        const std::string in = std::string("<function=Bash>\n<parameter=command>\necho ") +
                               s + "\n</parameter>\n</function>";
        const std::string out = q27::redact_drift(in);
        CHECK(out.find(s) == std::string::npos);
        // and no 8-byte run of it either, in case of partial copy
        for (size_t i = 0; i + 8 <= strlen(s); i++)
            CHECK(out.find(std::string(s + i, 8)) == std::string::npos);
    }
}
```

**Steps 2-5:** run (expect FAIL if any run leaks), fix `redact_drift` until it
passes, re-run, commit as `drift capture: leak gate -- no value byte survives`.

---

### Task 3: Shape hash and dedup key

**Files:**
- Modify: `src/drift_capture.h`, `tools/test_drift_capture.cpp`

**Dependencies:** Task 1

`shape_hash(redacted)` returns a stable 64-bit hash of the markup skeleton so
the corpus keeps one exemplar per distinct shape plus a count.

**Step 1: Write the failing test**

```cpp
static void test_shape_hash_ignores_values_not_structure() {
    const std::string a = "<function=Read>\n<parameter=file_path>\n/a\n</parameter>\n</function>";
    const std::string b = "<function=Read>\n<parameter=file_path>\n/b\n</parameter>\n</function>";
    const std::string c = "<function=Read>\n<parameter=file_path>\n/a\n</function>";  // closer dropped
    CHECK(q27::shape_hash(q27::redact_drift(a)) == q27::shape_hash(q27::redact_drift(b)));
    CHECK(q27::shape_hash(q27::redact_drift(a)) != q27::shape_hash(q27::redact_drift(c)));
}
```

**Steps 2-5:** implement (FNV-1a over the redacted string is sufficient — this
is a dedup key, not a security primitive), verify, commit.

---

### Task 4: Record format

**Files:**
- Modify: `src/drift_capture.h`, `tools/test_drift_capture.cpp`

**Dependencies:** Tasks 1, 3

`format_drift_record()` produces one JSON line: `id` (shape hash, hex),
`shape` (empty for now — labelled later), `tags` (`xml`/`json`,
`no_wrapper`, `in_think`), `outcome` (`recovered:<mode>` / `strict` /
`unrescued` / `suppressed`), `redacted` (the text), `bytes` (original length),
`ts`. JSONL, not NUL-separated: the old format cannot carry the labels, and
`third_party/json.hpp` is already vendored.

Test that the line parses as JSON, round-trips, and that `redacted` matches
`redact_drift` output exactly.

---

### Task 5: Widen the capture hook

**Files:**
- Modify: `src/api_common.h:5567-5580` (the existing hook)
- Modify: `src/api_common.h` — `parse_bare_tool_calls` success path

**Dependencies:** Task 4

**Step 1:** Replace the raw `fwrite` at the UN-RESCUED site with
`q27::write_drift_record(cp, ..., "unrescued")`.

**Step 2:** Add the same call on the success path (where `out` is non-empty)
with `"recovered:<mode>"`, and in `resolve_ordered_tool_segments` for the
`strict` case. Guard every one with `getenv("Q27_DRIFT_CORPUS")` so the
default build does nothing.

**Step 3:** Verify no behaviour change:

```bash
make test-tools
```
Expected: all suites pass, unchanged — this task must not alter a single
parse result.

**Step 4:** Verify capture works end to end:

```bash
Q27_DRIFT_CORPUS=/tmp/corpus.jsonl ./build/test_tool_drift_corpus
wc -l /tmp/corpus.jsonl && head -1 /tmp/corpus.jsonl | python3 -m json.tool | head
```
Expected: one JSONL record per dialect-bearing fixture, valid JSON, no raw
values present.

**Step 5:** Commit.

---

### Task 6: Corpus tooling

**Files:**
- Create: `tools/corpus_dedup.py`
- Create: `Makefile` target `corpus-dedup`

**Dependencies:** Task 5

Reads JSONL, groups by `id`, keeps one exemplar plus `count`, writes
`tools/drift_corpus/corpus.jsonl` sorted by descending count. Prints the shape
histogram — that histogram is the first look at what the model actually emits
in the wild, and is worth reading before any rewrite.

---

### Task 7: Wire into the fuzzer

**Files:**
- Modify: `Makefile` (the `fuzz` target)

**Dependencies:** Task 6

Copy `tools/drift_corpus/*` into `build/fuzz_corpus/` alongside
`tools/fuzz_seeds/*`. Real drift shapes are better fuzz seeds than synthetic
ones, and this closes the loop between the two efforts at zero cost.

---

## Definition of done

- `make test-tools` passes with no parse-result change (diff the before/after
  output of `test_tool_drift` and `test_openai_bridge` to prove it).
- The leak gate (Task 2) passes.
- A live serving session with `Q27_DRIFT_CORPUS=` set produces records whose
  `redacted` field contains no path, code or content byte from the session.
- `make corpus-dedup` prints a shape histogram.
- Default builds and default serving are byte-identical to before: every new
  code path is behind `getenv("Q27_DRIFT_CORPUS")`.

## Explicitly NOT in this phase

Labelling (`intended` / `constrained`), the strict parser, the rewrite
registry, and any change to parsing behaviour. Those are Phases 2-4.
