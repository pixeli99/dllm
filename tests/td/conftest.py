"""Make tests in tests/td/ runnable on machines that can't fully import dllm.

``dllm/__init__.py`` eagerly imports the entire pipelines tree, which
in turn needs ``lm_eval``, ``accelerate`` etc. On a stripped-down dev
machine those may be missing, so any ``from dllm.* import ...`` in a
test file fails at collection time even though the modules the tests
actually need are self-contained.

This conftest runs before pytest collects the test files. If a normal
``import dllm`` succeeds (cluster / full env), it does nothing. If not,
it installs minimal namespace packages for the relevant ``dllm.*``
prefixes and eagerly loads only the modules used by these tests --
schedulers, samplers (base / utils / mdlm_deterministic), llada_looped
cache_io, and trainers (td_losses / td_distill). Test files'
``from dllm.x.y import Z`` then resolves through the pre-loaded modules
rather than re-triggering ``dllm/__init__.py``.

This file is *only* a workaround for restricted test environments. On
the cluster the full dllm import path runs first and this conftest is a
no-op.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _full_dllm_import_works() -> bool:
    """Try the canonical import. Treat *any* exception as 'incomplete env'."""
    saved_modules = dict(sys.modules)
    try:
        importlib.import_module("dllm")
        return True
    except Exception:
        # Roll back any partial modules left in sys.modules so the
        # fallback below can install its own namespace packages cleanly.
        for k in list(sys.modules.keys()):
            if k == "dllm" or k.startswith("dllm."):
                if k not in saved_modules:
                    del sys.modules[k]
        return False


def _ensure_namespace(name: str) -> None:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m


def _eager_load(qualified_name: str, rel_path: str) -> None:
    spec = importlib.util.spec_from_file_location(
        qualified_name, os.path.join(_REPO_ROOT, rel_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build spec for {rel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)


if not _full_dllm_import_works():
    # Install minimal namespace stubs for the prefixes the tests touch.
    for ns in (
        "dllm",
        "dllm.core",
        "dllm.core.samplers",
        "dllm.core.schedulers",
        "dllm.core.trainers",
        "dllm.pipelines",
        "dllm.pipelines.llada_looped",
    ):
        _ensure_namespace(ns)

    # Load in dependency order. Each module only imports the ones above it.
    _eager_load("dllm.core.schedulers.alpha", "dllm/core/schedulers/alpha.py")
    _eager_load("dllm.core.schedulers.kappa", "dllm/core/schedulers/kappa.py")
    _eager_load("dllm.core.schedulers", "dllm/core/schedulers/__init__.py")

    _eager_load("dllm.core.samplers.base", "dllm/core/samplers/base.py")
    _eager_load("dllm.core.samplers.utils", "dllm/core/samplers/utils.py")
    _eager_load(
        "dllm.core.samplers.mdlm_deterministic",
        "dllm/core/samplers/mdlm_deterministic.py",
    )

    _eager_load(
        "dllm.pipelines.llada_looped.cache_io",
        "dllm/pipelines/llada_looped/cache_io.py",
    )

    _eager_load("dllm.core.trainers.td_losses", "dllm/core/trainers/td_losses.py")
    _eager_load("dllm.core.trainers.td_distill", "dllm/core/trainers/td_distill.py")
