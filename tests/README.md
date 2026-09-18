# Unit tests

Almost everything in this directory tests **gbench itself**. The one
exception is called out below.

Three things live in the repo and are easy to confuse, so they are kept
apart on purpose:

| | Unit tests (`tests/`) | Evals (`gbench/golden_dataset/`) | Template checks (`tests/test_chat_template_gemma4.py`) |
|---|---|---|---|
| Subject under test | the gbench code | whatever model you point gbench at | a Gemma 4 `chat_template.jinja` |
| Run by | `pytest` | `gbench --golden-only --remote-endpoint ...` | `GEMMA4_TEMPLATE_PATH=... pytest` |
| Needs a model? | no | yes | no, just the template file |
| Deterministic? | yes, always | no, that is the point | yes |
| Gates merges? | yes, via CI | no | no, skips unless you opt in |
| Ships in the wheel? | no | yes | no |

## Running them

```bash
pytest
```

No GPU, no network, no model weights. The Golden Set tests stand up a
fake in-process OpenAI-compatible endpoint with scripted answers, so
they assert that the runner *scores* correctly rather than that a model
*answers* correctly. A failure here means gbench is broken.

## Why the split matters

`gbench/golden_dataset/` is an eval: a deterministic smoke test over
basic capabilities, shipped so other repos can point `--golden-only` at
their own endpoint while developing a model. Its pass rate is a fact
about that model on that day, and it is expected to go red sometimes.
That is a signal to the team developing the model, not a reason to
block a gbench merge.

CI must therefore never run the eval. If it did, a bad checkpoint
somewhere else would turn this repo's `main` red, and the eval's real
audience would learn to ignore it.

The unit tests are the opposite: they must be green on every pull
request. `tests/test_golden.py` in particular carries the regression
guard for the defect that motivated this split. An adversarial fixture
that returns factually inverted and actively unsafe answers once scored
6/6 at 100% under substring matching. `test_adversarial_responses_all_fail`
now pins every one of those to `failed`.

## The chat-template checks

`tests/test_chat_template_gemma4.py` sits with the unit tests because it
is plain pytest, but it belongs to the eval side of that line: it judges
a template artifact, not gbench. So it follows the same rule and does
not gate merges. It resolves nothing by default and skips, and it never
downloads, because a module that fetches from Hugging Face on import
turns CI red whenever that host has a bad day and swaps the artifact
under you with no commit to point at.

```bash
curl -L -o /tmp/gemma4.jinja \
  https://huggingface.co/google/gemma-4-31B-it/resolve/main/chat_template.jinja
GEMMA4_TEMPLATE_PATH=/tmp/gemma4.jinja pytest tests/test_chat_template_gemma4.py
```

Point `$GEMMA4_TEMPLATE_PATH` at a candidate template to check a fix
before it ships. 29 checks assert what the shipped template already
does. 19 assert `google-deepmind/dialog` behaviour it does not, and are
`xfail(strict=True)`, so closing one of those gaps reports an XPASS and
the stale marker gets removed rather than quietly lying.
