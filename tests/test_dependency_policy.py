# -*- coding: utf-8 -*-
"""WS6: dependency policy. gbench never installs/clobbers the serving stack (vllm/torch/transformers);
it requires it at run time only for local serving. Eval extras stay additive to the pinned core."""

import inspect
import pathlib
import tomllib


def _opt_deps():
    root = pathlib.Path(__file__).resolve().parents[1]
    with open(root / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["optional-dependencies"]


def _names(reqs):
    out = []
    for r in reqs:
        n = r.split(";")[0].split("[")[0]
        for sep in ("<", ">", "=", "!", "~", " "):
            n = n.split(sep)[0]
        out.append(n.strip().lower())
    return out


def test_local_extra_does_not_install_the_serving_stack():
    opt = _opt_deps()
    assert "local" in opt, "keep the [local] extra so `pip install gbench[local]` still resolves"
    for pkg in ("vllm", "torch", "transformers"):
        assert pkg not in _names(opt["local"]), (
            f"{pkg} must NOT be auto-installed by gbench[local] (provisioned per docs; "
            "gbench hard-errors at run time if it is missing)")


def test_eval_extras_do_not_pull_the_serving_stack():
    opt = _opt_deps()
    for extra in ("evals", "omnidocbench", "ui"):
        got = _names(opt.get(extra, []))
        for pkg in ("vllm", "torch", "transformers"):
            assert pkg not in got, f"the [{extra}] extra must not pull {pkg} (keeps it additive)"


def test_require_vllm_message_points_to_provisioning_not_the_local_extra():
    from gbench.utils.dependency_checks import require_vllm_engine
    src = inspect.getsource(require_vllm_engine)
    assert "gbench[local]" not in src, "must not tell users to pip install gbench[local] for vllm"
    assert "pip install vllm" in src
    assert "remote-endpoint" in src, "should note remote runs do not need vllm"
