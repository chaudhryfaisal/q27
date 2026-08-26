# Parser: normalize, then parse strictly

2026-08-24. Design for replacing the ad-hoc drift-recovery chain with a
normalization stage plus one strict parser. Written after a week in which the
parser changed four times.

## Why

99 commits touched the parser in 60 days; `src/api_common.h` alone took
+6,694 / -581 lines. Classifying the last month's fixes: **19 were "a shape we
had not enumerated", 2 were "a channel we had not covered."** The catalogue is
an open set. Each fix buys exactly one shape and adds surface.

The root cause is measurable. In the Qwen3.8 tokenizer:

| framing | tokenization |
|---|---|
| `<think>` / `</think>` | 1 atomic special token (248068 / 248069) |
| `<\|im_start\|>` / `<\|im_end\|>` | 1 atomic special token |
| `<tool_call>` | 4 ordinary text tokens |
| `<function=` / `<parameter=` | 3 ordinary text tokens |

Thinking and turn boundaries are **out-of-band**; tool framing is **in-band**.
The splitter never guesses about `</think>` because the model cannot spell it
accidentally. It must always guess about calls, because `<tool_call>` is just
text the model can write about, drop, nest, or truncate. Sixteen recovery
modes exist to guess intent after the fact.

Out-of-band framing is the structurally correct fix and is unavailable: it
needs a model trained to emit a reserved tool-call token, and Qwen3.8 was not.
Recorded here as the thing to want if we ever train.

## The change

The chain already splits along a line the code implies: ~140 sites are
string-to-string rewrites (synthesize a dropped opener, blank stray closers,
repair braces), ~17 are semantic inference (infer the tool name, canonicalize
it, judge intent). Separate them.

```
generated text
   |  normalization: N small pure rewrites, each independently testable
canonical form: <tool_call>{"name":...,"arguments":{...}}</tool_call>
   |  ONE strict parser (small, fuzzable, no recovery logic)
ToolCall
   |  semantic completion: name inference, canonicalization (explicit, separate)
```

A new drift shape becomes **a rewrite rule, never a new parse path**. The
strict parser stops growing, which is what makes the fuzz result
(docs/SECURITY.md) durable rather than a snapshot of one afternoon.

**Unchanged:** the splitter and its channels, the display-context lexer
(`markdown_lex.h`), `--constrain-tools`, and the streaming holdback's job.
None of those are where the churn is.

**Non-goals.** This does not make constrained decoding the default -- that is
a separate bet with its own serving-quality risk and needs its own
measurement. It does not touch the sampler. It does not attempt out-of-band
framing.

## The corpus and its oracle

Migration is corpus-first: build the oracle before the rewrite.

**Capture needs widening.** `Q27_DRIFT_CORPUS` writes only inside the
UN-RESCUED branch today -- misses only. A regression oracle needs the
successes, since those are what a rewrite must not break. Capture moves to
every generated turn containing dialect markup, tagged with what the chain did:
`recovered:<mode>`, `strict`, `unrescued`, `suppressed`.

**Redaction happens at capture time**, before anything reaches disk. q27 is a
public repo and the corpus is real session content.

- Values (XML: between `<parameter=k>` and its closer; JSON: string values)
  become typed placeholders `PATH_1`, `CODE_2`, `TEXT_3`. Type from the key,
  index stable within an entry.
- **Preserved inside values:** any dialect markup -- a `</function>` inside a
  value is structure, and is the exact case that broke the closer-bounding fix
  -- plus length class (short / multiline / >4 KB) and leading/trailing
  whitespace shape.
- Keys, tool names and all framing survive verbatim. That is the grammar.

**Three-way label per entry**, as a sidecar:

```json
{ "id":"...", "shape":"bare_function_in_think", "tags":["xml","no_wrapper"],
  "intended":    {"calls":[{"name":"Read","arguments":{"file_path":"PATH_1"}}]},
  "current":     {"calls":[], "mode":17},
  "constrained": {"calls":[], "note":"disengaged at byte 0x21"} }
```

`intended` is human-decided and is the actual oracle. `current` is recorded
automatically and is the regression baseline. `constrained` is recorded by
replaying under `--constrain-tools`, which makes the corpus reusable for that
separate bet without re-capturing.

Dedup by shape hash (the markup skeleton after redaction), keeping one
exemplar plus a count, so the committed set stays small while real frequency
stays visible.

## The rewrite-rule contract

Every rule is a value, not a branch:

```cpp
struct Rewrite {
  const char* name;              // "synthesize_dropped_opener"
  const char* provenance;        // "issue #24, 2026-08-21"
  bool (*applies)(const Ctx&);   // pure predicate over text + display context
  void   (*apply)(Buf&);         // pure edit, records an offset mapping
  const char* fires_on;          // input it MUST rewrite
  const char* never_fires_on;    // input it MUST NOT touch
};
```

**Display context is checked once, centrally.** Today every mode must remember
to consult `markdown_lex.h`, and forgetting is how a fenced example becomes an
executed call. The driver evaluates executability at each candidate position
and only then offers it to `applies()`, so a rule *cannot* fire inside a fence
or a backtick span even if its author forgot. This converts the sharpest
security invariant -- a model writing ABOUT a call must not make one -- from a
convention into a structural property.

**Offset mapping is mandatory**, and it constrains everything.
`resolve_ordered_tool_segments` needs `source_begin`/`source_end` in the
ORIGINAL text to emit surrounding prose, and the `SIZE_MAX` underflow class
lives exactly there. `Buf` carries an edit log; canonical offsets map back to
original spans. Rewrites that cannot be mapped (pure synthesis) mark a
zero-width origin rather than lying about one.

**Ordering:** declared order, at most 3 bounded passes, each rule idempotent.
No fixpoint -- arbitrary rules plus fixpoint is non-termination on adversarial
input, which is a fuzz target's favourite discovery.

**Semantic completion stays out.** `infer_tool_name`,
`canonical_declared_name` and intent judgement run after the strict parse, on
structured data, where they are testable without string handling.

`fires_on` / `never_fires_on` generate tests and fuzz seeds automatically. The
registry becomes the thing that was missing while 19 shapes arrived one at a
time.

## Phasing

**Phase 1 -- capture (ships alone, useful regardless).** Widen
`Q27_DRIFT_CORPUS`, add capture-time redaction, shape hash, label sidecar. No
parser change. Gate: existing 698 assertions pass; a test asserts no captured
entry contains a byte from the source values.

**Phase 2 -- the oracle.** Replay under `--constrain-tools` to fill
`constrained`. Hand-label `intended`: misses first, then a spot-check pass over
successes, because the point of a three-way label is to catch where `current`
is confidently wrong. Deliverable: `make corpus-check` reporting
current-vs-intended agreement. **That number is the first real finding.**

**Phase 3 -- strict parser + registry, in shadow.** Both run alongside the
existing chain, diffing on every request, no behaviour change. Gate: zero
divergence across the corpus and 7 days of live traffic.

**Phase 4 -- switch.** Old chain behind `Q27_PARSER=v1` for one release.

## Success criteria, stated so they can fail

- Strict parser <= 400 lines, and does not grow when a new shape arrives.
- A new drift shape is one registry entry plus two literals; no parse-code change.
- `make fuzz` covers the strict parser and every rule's `never_fires_on`.
- Tool-call parity holds at 1.0000 with PART <= the current 1/597.
- Parser-file line delta per new shape drops from ~340 (this month's average)
  to under 20.

## Kill criteria

If Phase 2 shows `current` and `intended` agree on more than 99% of entries,
the catalogue is nearly right, the rewrite buys only churn reduction, Phase 3
is optional, and the cheaper harness-only option was the correct call. Say so
and stop.
