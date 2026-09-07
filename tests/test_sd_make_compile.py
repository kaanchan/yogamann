"""
Regression test for #14: torch.compile must never be silently skipped when
Triton is unavailable (sm_120/Blackwell has no stable Triton yet) — it should
fall back to the Triton-free cudagraphs backend instead.
"""
import sys
import types

import pytest

import sd_make


def test_compile_unet_falls_back_to_cudagraphs_without_triton(monkeypatch):
    # This environment genuinely has no triton installed (sm_120 not yet
    # supported) — exercises the real ImportError path, not a simulation.
    monkeypatch.delitem(sys.modules, "triton", raising=False)

    calls = []
    monkeypatch.setattr(
        sd_make.torch, "compile", lambda unet, **kw: calls.append(kw) or unet
    )

    unet = object()
    result = sd_make._compile_unet(unet)

    assert result is unet
    assert calls == [{"backend": "cudagraphs"}]


def test_compile_unet_uses_reduce_overhead_when_triton_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))

    calls = []
    monkeypatch.setattr(
        sd_make.torch, "compile", lambda unet, **kw: calls.append(kw) or unet
    )

    unet = object()
    result = sd_make._compile_unet(unet)

    assert result is unet
    assert calls == [{"mode": "reduce-overhead", "fullgraph": False}]
