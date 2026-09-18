# Scaffold versioning

## Why

A benchmark score is only interpretable if the model *and* the harness stayed
fixed. If the prompt template, the few-shot examples, the output schema, the
decode settings, the scoring script or the tool setup moved between two runs,
a score change no longer tells you whether the model improved or regressed. It
tells you that something moved, and you have no way to say what.

So gbench treats the scaffold as part of the experiment, roughly as important
as the weights. It is versioned, named, and printed beside every result.

> Pin the scaffold as tightly as the weights, otherwise your benchmark is
> measuring a moving target.

## The eleven fields

| Field | Axis |
| --- | --- |
| `model_checkpoint` | subject |
| `quantization` | subject |
| `prompt_version` | scaffold |
| `schema_version` | scaffold |
| `scoring_version` | scaffold |
| `tool_permissions` | scaffold |
| `attempts` | scaffold |
| `decode` | scaffold |
| `dataset_snapshot` | scaffold |
| `serving_environment` | scaffold |
| `load_shape` | scaffold |

`serving_environment` and `load_shape` are scaffold fields for the same
reason the prompt is. Serving the same weights under a different engine,
a different tensor-parallel width or a different KV budget is a different
question, and `batch_sizes=[1, 16, 50, 100]` is a property of the harness
rather than of the model.

## Two ids, not one

A run is the pair `(subject, scaffold)`, and the two are hashed separately.

- `subject_id`: what was tested. Model id, serving format, GGUF file.
- `scaffold_id`: how we asked. Everything else.

The separation is the entire design, and it exists to make one specific
comparison legal:

| | valid? | reading |
| --- | --- | --- |
| `subject_id` moved, `scaffold_id` held | yes | a like-for-like model comparison |
| `scaffold_id` moved, `subject_id` held | yes | a new experimental condition |
| both moved | no | uninterpretable, and the dashboard should not join them |

A quantization sweep is the first row: several subjects measured against one
held scaffold. A single combined hash would make every subject produce a
different id and there would be nothing left to hold constant, so the sweep
could not be expressed at all.

## Reading it from the CLI

The console and `metadata.json` are two views of the same objects. Neither is
required for the other to work, because plenty of gbench runs are headless and
some consumers only ever use the CLI.

```
$ gbench --models gemma-3-4b-it --golden-only

================================================================================
SCAFFOLD CONTRACT (v1)
================================================================================

A result is the pair (subject, scaffold). Holding scaffold_id while
subject_id moves is a like-for-like model comparison. A scaffold_id that
moved means the harness moved, so scores either side are not comparable.

PILLAR  MODEL          FORMAT  SUBJECT      SCAFFOLD
golden  gemma-3-4b-it  hf      sb_f421545a  sc_71d8150b
golden  gemma-3-4b-it  gguf    sb_cc68399f  sc_71d8150b

sc_71d8150b  pins 8/11: attempts, dataset_snapshot, decode, prompt_version,
             schema_version, scoring_version, serving_environment,
             tool_permissions
             dataset gd_661165ef over 16 case(s)
             unpinned: model_checkpoint, quantization, load_shape
```

Two subjects, one scaffold. That is the sweep, and it is readable without a
service, an upload or a browser.

The same conditions are written to `metadata.json` under `conditions`, one
entry per `(pillar, model, format)` cell of the run's matrix.

## Coverage is disclosed, not assumed

`covers` and `unpinned` are not decoration. An id that silently claimed to pin
nine fields while actually pinning six would be worse than no id, because it
would licence a comparison that is not sound.

`unpinned` is derived from `covers` rather than hand-maintained, so the two
cannot drift, and `scaffold_id` hashes exactly the fields named in `covers`.
It is structurally impossible for the id to over-promise.

Note the direction of the guarantee. Disclosure makes the id *honest about its
scope*. It does not *detect* a change in an unpinned field. Only a content
hash detects.

### Why golden pins eight and not eleven

- `model_checkpoint` and `quantization` are the subject axis, deliberately
  excluded so the sweep above works.
- `load_shape` is unpinned, and that is not an oversight. Golden issues its
  cases one at a time, so it does have a concurrency, it is one, and nothing
  in the harness records that as a decision. When it becomes explicit it
  should move into `covers`.

The other eight are pinned, three of them by subsumption: `prompt_version`,
`schema_version` and `tool_permissions` all live inside the case JSON, so a
content hash over `gbench/golden_dataset/*.json` covers all three at once.
Exactly three of the sixteen cases declare top-level `tools`/`tool_choice`.

A pillar whose scaffold is not yet modelled reports `scaffold_id: null` and
`0/11` rather than being omitted, so the gap is visible instead of looking
like the pillar did not run.

## The scorer

`scoring_version` means different things in the two pillars that pin it, and
both readings are deliberate.

Golden's scorer is code in `gbench/runners/golden.py`, not data, so it pins a
declared integer `SCORING_VERSION` that a human bumps when grading changes.
The alternative, hashing the source file, would break every series on an
unrelated edit to that module, and an id that churns on changes touching
nothing relevant stops carrying information.

Quality's scorer is GemmaClaw, a separate repository, so it pins the commit.
`--gemmaclaw-commit` defaults to a sha rather than to a branch, because a
branch name is not a pin: it names a different commit next week while the
hash of the string `"main"` never moves. The default lives in
`DEFAULT_GEMMACLAW_COMMIT` in `gbench/core/config.py` and is currently
`d8bc6989...`, the `gemmaclaw-v2026.8.3` release.

```
$ gbench --models gemma-3-1b-it --quality-only --dry-run

PILLAR   MODEL          FORMAT  SUBJECT      SCAFFOLD
quality  gemma-3-1b-it  hf      sb_d11894fd  sc_27c5eb72
quality  gemma-3-1b-it  gguf    sb_f9285816  sc_27c5eb72
```

So the same bare command gives the same `scaffold_id` in six months. Nothing
is looked up and the run does not need GitHub to be reachable.

Anything that is not already a sha is resolved before it is hashed. Ask for
the development tip and the id moves to say the scorer moved:

```
$ gbench --models gemma-3-1b-it --quality-only --dry-run --gemmaclaw-commit main
INFO - Resolved gemmaclaw ref 'main' to 08c0584d660591fa713928df9249e4ec37322ea5

PILLAR   MODEL          FORMAT  SUBJECT      SCAFFOLD
quality  gemma-3-1b-it  hf      sb_d11894fd  sc_a582b40b
quality  gemma-3-1b-it  gguf    sb_f9285816  sc_a582b40b
```

Resolving once rather than per condition is what keeps the sha in the id and
the sha in the checkout the same commit. Two lookups could straddle a push to
`main` and pin a scorer the run never used.

A ref that cannot be resolved is left as the literal string. That is honest
about the gap rather than a guess, and it predicts a failed run anyway,
because a ref the resolver cannot resolve is a ref the runner cannot check
out.

### Promoting the pinned scorer

By hand, and deliberately so. Take the newest `gemmaclaw-v*` tag, put its sha
in `DEFAULT_GEMMACLAW_COMMIT`, update the tag name and date in the comment
above it, and say so in the PR. There is no script for this.

Automating it would defeat the pin. A default that discovers the newest
release on its own is still a moving target, just a slower one: the day a new
tag lands, every unflagged run silently changes scorer. Promotion is supposed
to be a decision someone made and a reviewer saw.

## The dataset snapshot

`dataset_snapshot` hashes the canonical form of every case, plus every binary
asset by content. Canonical means sorted keys, no incidental whitespace, and
integers normalised to floats, so reformatting a case cannot move the id but
changing a prompt does.

All assets are hashed even on a subset run. Attributing an image to the cases
that reference it would mean parsing every message, and over-detecting is the
safe direction here: a spurious break costs a glance at a diff, a missed one
costs a wrong conclusion.

## The academic evals pillar

Golden and Quality above pin their scorer directly. The academic evals pillar
(`--evals`) pins its scorer version indirectly, through the per-suite
canonical-sync registry (`gbench/runners/eval_suites/canonical_sync.py`):

- Each runnable suite has an entry recording `synced` (the date it was last
  reconciled against upstream), and optionally `revision` (the pinned upstream
  commit/tag) and `checksum` (a dataset/upstream content fingerprint). The
  dispatcher stamps this onto every result, and `gbench --eval-provenance`
  prints the table (marking each entry version-pinned or date-only).
- The evals scaffold folds each selected suite's `synced` + `revision`/`checksum`
  into its `scoring_version` component. So when a suite's scorer, prompt,
  extraction, or dataset changes, bumping its canonical-sync entry MOVES the
  evals `scaffold_id`, which correctly marks results either side of the change
  as not comparable. Without the bump the scaffold would not move and two
  non-comparable runs would look comparable.

### Updating a suite

When you change a suite's scorer / prompt / extraction / dataset:

1. Make the code change.
2. Reconcile it against the upstream benchmark (do not invent a date or commit).
3. Update that suite's `canonical_sync.py` entry: set `synced` to today, and set
   `revision`/`checksum` when you pinned a specific upstream version or dataset
   snapshot. This is what moves the scaffold and records the provenance.

### The gate (recommended CI)

A CI check should fail when a `gbench/runners/eval_suites/*.py` diff touches a
scorer/prompt/extraction and the same commit does not bump that suite's
canonical-sync `synced`/`revision`, mirroring the `Golden Changelog` gate. Until
that gate lands, treat step 3 as part of the definition of done for any eval
scorer change.

## Changing a case

A snapshot id that moved tells a reader that the dataset changed. It does not
tell them *why*, and six months later "`gd_661165ef` became `gd_9c1e04b2`" is
most of the way to no information at all.

So a case whose content changed needs two things:

1. a bump to `meta.version`
2. an appended `meta.changelog` entry for the new version, with a non-empty
   `reason`

```json
"meta": {
  "version": "1.2",
  "changelog": [
    {
      "version": "1.1",
      "reason": "Baseline entry recorded when the scaffold versioning contract was adopted. Changes before this point are in git history rather than here."
    },
    {
      "version": "1.2",
      "reason": "Tightened answer_pattern so a longer number such as 4200 no longer satisfies a golden of 42."
    }
  ],
  "description": "..."
}
```

The changelog is append-only. Past entries may not be edited or removed, so
history cannot be quietly rewritten.

`meta.changelog` is excluded from the hash and from the gate's comparison.
Without that exclusion the rule is circular: appending the entry that explains
a bump is itself a content change, which would demand another bump, which
would demand another entry.

### The gate

```
$ python scripts/check_golden_changelog.py --base origin/main
```

Runs in CI as the `Golden Changelog` check. Reformatting alone is not a
content change and needs neither a bump nor an entry, because the gate reads
the same `canonical_json` the snapshot id reads. That equivalence is the point:
a change that would not move the id cannot trip the gate, and a gate that
nags on reindentation is one that gets switched off.

Deleting a case is reported rather than failed. There is no file left to carry
an entry, so the rule is unenforceable by construction.

## Where the code lives

| File | Role |
| --- | --- |
| `gbench/canonical.py` | `canonical_json`, `strip_changelog`. Stdlib only. |
| `gbench/core/scaffold.py` | Contract objects, id construction, console rendering. |
| `gbench/changelog_gate.py` | The changelog rules. |
| `scripts/check_golden_changelog.py` | Git plumbing for the gate. |
| `.github/workflows/golden-changelog.yml` | CI wiring. Needs `fetch-depth: 0`. |

`canonical.py` and `changelog_gate.py` sit outside `gbench.core` on purpose.
`gbench/core/__init__.py` imports the model registry, which contacts Hugging
Face at import time. The gate reads JSON out of a git diff and has no business
touching the network, so it imports clean with no packages installed at all. A
policy gate that goes red when the Hub is slow is a policy gate that gets
marked non-required.

## Contract version

`CONTRACT_VERSION` is bumped when the *meaning* of the blocks changes in a way
that makes older ids incomparable with newer ones. Adding a field to `covers`
does not need a bump, because `covers` already tells the reader what moved.

The gbench commit is recorded beside the ids and never inside them. As an
input it would invalidate every scaffold on every commit to this repo, and an
id that churns on changes touching nothing relevant stops carrying
information.
