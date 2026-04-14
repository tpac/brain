# Change Documentation Standard

**This is how brain/claude documents proposed and implemented changes.**

When you discuss, plan, or implement a significant change to the brain's pipeline (encoding, recall, scoring, hooks, voice, telemetry), write a change document in `docs/` using this structure. The goal: any fresh Claude session can find this document, understand what changed, WHY it changed, what risks exist, and how to test it.

---

## File Naming

```
docs/{feature-name}-{date}.md
```

Examples:
- `docs/encoding-decoding-v2-2026-03-23.md`
- `docs/ripple-engine-2026-03-25.md`
- `docs/precision-scorer-v3-2026-04-01.md`

---

## Required Sections

### 1. Header

```markdown
# {Feature Name}

**Date:** YYYY-MM-DD
**Git version:** {commit hash} ({branch})
**Status:** PROPOSED | IN PROGRESS | SHIPPED | REVERTED
**Author:** Claude {model} + {operator name}
**Session:** #{number}
**Benchmark baseline:** NDCG={x}, MRR={y}, passed={z}/{total} @ commit {hash}
```

The git version is the commit at the time the document was CREATED. If the change ships, add a "Shipped at" line with the final commit.

### 2. Why This Change

One paragraph. What problem does this solve? What failure did we observe? Link to the benchmark data or golden dataset failures that motivated it.

**Include:**
- The specific failure mode (e.g., "semantic category: 0/16 passing")
- What we tried that DIDN'T work (and why — prevents future sessions from re-trying)
- What the operator said that triggered this direction (quote if possible)

### 3. Flow Diagrams

Use the box-drawing format established in `encoding-decoding-v2-2026-03-23.md`:

```
═══════════════════════════════════════════════════════════════
{SECTION TITLE}
Files: {file.py} → {function()}
Models: {model name} ({role})
Tables: {table names, ★NEW for new ones}
═══════════════════════════════════════════════════════════════

   {Entry point}
        │
        ▼
   {file.py} → {function()}
   ├─ {step description}
   │   └─ {detail}
   ├─ ★NEW: {new step}
   └─ Return {what}
```

**Always include:**
- File names and function names (so a fresh Claude can `grep` for them)
- Model names and their roles (Arctic v1.5 for embedding, BART for keywords, etc.)
- Table names (nodes, node_embeddings, node_enrichments, edges, etc.)
- Which steps are NEW vs unchanged
- Concrete examples with real data (not abstract descriptions)

**Write three flows when proposing a change:**
1. **OLD** — how it worked before
2. **CURRENT** — how it works now
3. **PROPOSED** — how it will work after this change

### 4. Risks and Sacred Systems

```markdown
## Risks

**Sacred systems touched:**
- [ ] Embedding pipeline (servers/embedder.py)
- [ ] Recall pipeline (servers/brain_recall.py, recall_scorer.py)
- [ ] Encoding pipeline (servers/brain_remember.py, brain_engineering.py)
- [ ] Precision pipeline (servers/brain_precision.py)
- [ ] Hook output format (servers/brain_voice.py wrap_for_hook())

**What could break:**
- {Specific failure mode}: {what triggers it, what the user sees}
- {Specific failure mode}: {what triggers it, what the user sees}

**Rollback plan:**
- {How to revert if this breaks production}

**Silent failure risk:**
- {Any path where an error could be swallowed without logging}
- RULE: No bare `except: pass`. Every exception must log to brain_logs.db.
```

### 5. Testing Strategy

```markdown
## Testing

**Before writing ANY code:**
1. Run golden dataset baseline:
   ```
   BRAIN_DB_DIR=$HOME/AgentsContext/brain python3 tests/eval_runner.py $HOME/AgentsContext/brain/brain.db
   ```
2. Record: NDCG={x}, MRR={y}, passed={z}/{total}
3. Save this as the "before" number

**Test cases to add to golden_dataset.json:**
- {Category}: "{query}" should find "{expected node title}" — tests {what this proves}
- {Category}: "{query}" should NOT outrank "{other node}" — tests {regression guard}

**After implementation:**
1. Run golden dataset again — NDCG must not drop
2. Run E2E tests: `python3 -m pytest tests/test_e2e_enrichment.py -v`
3. Run full test suite: `python3 tests/run_tests.py --golden`
4. {Any manual testing steps}

**Variations tested:**
| Variant | NDCG | MRR | Passed | Verdict |
|---|---|---|---|---|
| {variant A} | | | | |
| {variant B} | | | | |
```

### 6. Implementation Checklist

```markdown
## Implementation

- [ ] Benchmark baseline recorded at commit {hash}
- [ ] New golden dataset cases added (if applicable)
- [ ] Code changes (list files)
- [ ] Tests written
- [ ] Tests passing
- [ ] Benchmark re-run — no regression
- [ ] Brain node encoded with findings
- [ ] Document status updated to SHIPPED
- [ ] Shipped at commit {hash}
```

### 7. Operator Notes

Anything the operator (Tom) said during the session that shaped the decision. Direct quotes are best. These are the WHYs that get lost between sessions.

```markdown
## Operator Notes

> "dont silent kill an exception, we need to know what works and what doesnt"
— Drives the telemetry requirement

> "LLMs need as much info as possible, what is Glo is wrong, we need much more content"
— Why HyDE with bare queries failed; led to structured prompts with neighbor context

> "encoding does some decoding before encoding, thats really how the brain works"
— The insight behind recall-before-encode and ripple architecture
```

---

## Where to Find Change Documents

All change documents live in `docs/`. A fresh Claude session should:

1. `ls docs/` to see all change documents
2. Read the most recent ones for current state
3. Check `Status:` header — PROPOSED means not yet built, SHIPPED means in production
4. Check `Git version:` — if the repo has moved far past the document's commit, the document may be stale

---

## Encoding to Brain

After writing a change document, encode a brain node pointing to it:

```
remember(
  type="decision",
  title="Change doc: {feature name} ({date})",
  content="See docs/{filename}. {One sentence summary of what changed and why}.",
  keywords="{relevant keywords}",
  locked=True
)
```

The brain node holds the WHY and the POINTER. The document holds the WHAT and HOW.
