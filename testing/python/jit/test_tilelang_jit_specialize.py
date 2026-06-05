from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

import tilelang
import tilelang.language as T
from tilelang.jit import JITImpl


class TensorLike:

    def __init__(self, *shape):
        self.shape = shape


class FakeKernel:

    def __init__(self, name):
        self.name = name
        self._tilelang_cache_key = f"fake-{name}"

    def __call__(self, *args):
        return self.name, args


def _specialized_lazy_kernel():
    m = tilelang.arg("A").shape[0]

    @tilelang.jit
    @tilelang.specialize(
        block_m=tilelang.bucket(m, [16, 64, 128], policy="ceil", overflow="error"),
        block_n=128,
        require=[m > 0],
        compile="on_miss",
    )
    def kernel(A, *, block_m, block_n):
        @T.prim_func
        def main():
            T.evaluate(block_m + block_n)

        return main

    return kernel


@pytest.fixture
def no_frontend_cache(monkeypatch):
    monkeypatch.setattr("tilelang.cache.load_frontend_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr("tilelang.cache.store_frontend_cache", lambda *args, **kwargs: None)


def test_public_specialization_api_imports_and_builds_spec():
    m = tilelang.arg("A").shape[0]

    @tilelang.specialize(
        block_m=tilelang.bucket(m, [16, 64, 128], policy="ceil", overflow="error"),
        block_n=128,
        block_k=64,
        require=[m > 0],
    )
    def fn(A, *, block_m, block_n, block_k):
        return A, block_m, block_n, block_k

    spec = fn.__tilelang_specialization__
    result = spec.evaluate({"A": TensorLike(17, 32)})

    assert tilelang.arg is not None
    assert tilelang.bucket is not None
    assert tilelang.specialize is not None
    assert result.as_kwargs == {"block_k": 64, "block_m": 64, "block_n": 128}
    assert result.cache_key == (("block_k", 64), ("block_m", 64), ("block_n", 128))


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((1, 8), 16),
        ((16, 8), 16),
        ((17, 8), 64),
        ((65, 8), 128),
    ],
)
def test_bucket_evaluation_maps_runtime_shapes(shape, expected):
    m = tilelang.arg("A").shape[0]
    spec = tilelang.specialize(
        block_m=tilelang.bucket(m, [16, 64, 128], policy="ceil", overflow="error"),
    )(lambda A, *, block_m: None).__tilelang_specialization__

    assert spec.evaluate({"A": TensorLike(*shape)}).as_kwargs == {"block_m": expected}


def test_specialization_validation_rejects_invalid_specs():
    m = tilelang.arg("A").shape[0]

    with pytest.raises(ValueError, match="non-empty"):
        tilelang.arg("")
    with pytest.raises(ValueError, match="at least one bucket"):
        tilelang.bucket(m, [])
    with pytest.raises(ValueError, match="strictly increasing"):
        tilelang.bucket(m, [16, 16, 64])
    with pytest.raises(ValueError, match="Unsupported bucket policy"):
        tilelang.bucket(m, [16, 64], policy="nearest")
    with pytest.raises(ValueError, match="ahead-of-time dispatch modes are future work"):
        tilelang.specialize(block_m=tilelang.bucket(m, [16]), compile="ahead_host_dispatch")
    with pytest.raises(ValueError, match="Callable specialization selectors"):
        tilelang.specialize(block_m=lambda A: 16)
    with pytest.raises(ValueError, match="requirements must be metadata comparison expressions"):
        tilelang.specialize(block_m=16, require=[True])
    with pytest.raises(ValueError, match="Metadata expressions must be wrapped"):
        tilelang.specialize(block_m=m)
    with pytest.raises(ValueError, match="stable scalars"):
        tilelang.specialize(block_m=[16])


def test_specialization_evaluation_rejects_runtime_metadata_errors():
    m = tilelang.arg("A").shape[0]
    dim3 = tilelang.arg("A").shape[3]
    spec = tilelang.specialize(
        block_m=tilelang.bucket(m, [16, 64, 128], policy="ceil", overflow="error"),
    )(lambda A, *, block_m: None).__tilelang_specialization__
    dim_spec = tilelang.specialize(block_m=tilelang.bucket(dim3, [16]))(
        lambda A, *, block_m: None
    ).__tilelang_specialization__

    with pytest.raises(ValueError, match="exceeds largest bucket"):
        spec.evaluate({"A": TensorLike(129, 8)})
    with pytest.raises(ValueError, match="Missing runtime argument.*A"):
        spec.evaluate({})
    with pytest.raises(ValueError, match="no shape dimension 3"):
        dim_spec.evaluate({"A": TensorLike(1, 2)})


def test_require_checks_pass_and_fail_before_compile(monkeypatch, no_frontend_cache):
    m = tilelang.arg("A").shape[0]
    n_a = tilelang.arg("A").shape[1]
    n_b = tilelang.arg("B").shape[1]

    @tilelang.jit
    @tilelang.specialize(block_m=tilelang.bucket(m, [16, 64]), require=[m > 0, n_a == n_b])
    def kernel(A, B, *, block_m):
        @T.prim_func
        def main():
            T.evaluate(block_m)

        return main

    def fake_compile(self, *args, **kwargs):
        return FakeKernel(f"block-{kwargs['block_m']}")

    monkeypatch.setattr(JITImpl, "compile", fake_compile)

    assert kernel(TensorLike(1, 4), TensorLike(8, 4)).name == "block-16"
    with pytest.raises(ValueError, match="requirement failed"):
        kernel(TensorLike(0, 4), TensorLike(8, 4))
    with pytest.raises(ValueError, match="requirement failed"):
        kernel(TensorLike(1, 4), TensorLike(8, 5))


def test_specialized_jit_reuses_variants_by_bucket(monkeypatch, no_frontend_cache):
    compile_kwargs = []
    kernel = _specialized_lazy_kernel()

    def fake_compile(self, *args, **kwargs):
        compile_kwargs.append(dict(kwargs))
        return FakeKernel(f"compile-{len(compile_kwargs)}")

    monkeypatch.setattr(JITImpl, "compile", fake_compile)

    first = kernel(TensorLike(1, 8))
    second = kernel(TensorLike(2, 8))
    third = kernel(TensorLike(33, 8))
    fourth = kernel(TensorLike(40, 8))

    assert first is second
    assert third is fourth
    assert first is not third
    assert [kwargs["block_m"] for kwargs in compile_kwargs] == [16, 64]
    assert all("block_m" not in key_part for cache_key in kernel._kernel_cache for key_part in cache_key[:1])


def test_specialization_rejects_conflicting_user_compile_kwarg(monkeypatch, no_frontend_cache):
    kernel = _specialized_lazy_kernel()

    def fail_compile(self, *args, **kwargs):
        raise AssertionError("conflict must fail before compile")

    monkeypatch.setattr(JITImpl, "compile", fail_compile)

    with pytest.raises(ValueError, match="selected `block_m=16`"):
        kernel(TensorLike(1, 8), block_m=64)


def test_specialization_overrides_default_compile_kwarg(monkeypatch, no_frontend_cache):
    m = tilelang.arg("A").shape[0]
    compile_kwargs = []

    @tilelang.jit
    @tilelang.specialize(block_m=tilelang.bucket(m, [16, 64]))
    def kernel(A, *, block_m=16):
        @T.prim_func
        def main():
            T.evaluate(block_m)

        return main

    def fake_compile(self, *args, **kwargs):
        compile_kwargs.append(dict(kwargs))
        return FakeKernel(f"block-{kwargs['block_m']}")

    monkeypatch.setattr(JITImpl, "compile", fake_compile)

    assert kernel(TensorLike(33, 8)).name == "block-64"
    assert compile_kwargs == [{"block_m": 64}]


def test_specialized_frontend_cache_key_includes_variant(monkeypatch):
    loads = []
    sentinel = FakeKernel("frontend")
    kernel = _specialized_lazy_kernel()

    def fake_load_frontend_cached(frontend_key_data, **kwargs):
        loads.append(frontend_key_data)
        return sentinel

    def fail_compile(self, *args, **kwargs):
        raise AssertionError("frontend cache hit should not compile")

    monkeypatch.setattr("tilelang.cache.load_frontend_cached", fake_load_frontend_cached)
    monkeypatch.setattr(JITImpl, "compile", fail_compile)

    assert kernel(TensorLike(1, 8)) is sentinel
    assert loads[0]["specialization"] == (("block_m", 16), ("block_n", 128))


def test_compile_failure_does_not_poison_specialized_cache(monkeypatch, no_frontend_cache):
    kernel = _specialized_lazy_kernel()
    attempts = 0

    def flaky_compile(self, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary compile failure")
        return FakeKernel("retry")

    monkeypatch.setattr(JITImpl, "compile", flaky_compile)

    with pytest.raises(RuntimeError, match="temporary compile failure"):
        kernel(TensorLike(1, 8))
    assert not kernel._kernel_cache
    assert kernel(TensorLike(2, 8)).name == "retry"
    assert attempts == 2


def test_cold_variant_compiles_once_with_concurrent_callers(monkeypatch, no_frontend_cache):
    kernel = _specialized_lazy_kernel()
    compile_calls = 0
    compile_lock = threading.Lock()

    def slow_compile(self, *args, **kwargs):
        nonlocal compile_calls
        time.sleep(0.05)
        with compile_lock:
            compile_calls += 1
            call_id = compile_calls
        return FakeKernel(f"compile-{call_id}")

    monkeypatch.setattr(JITImpl, "compile", slow_compile)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda shape: kernel(TensorLike(shape, 8)), [33, 34, 35, 36]))

    assert compile_calls == 1
    assert len({id(result) for result in results}) == 1


def test_unspecialized_jit_does_not_get_specialization_metadata(monkeypatch, no_frontend_cache):
    compile_kwargs = []

    @tilelang.jit
    def plain_kernel(block_m: int = 16):
        @T.prim_func
        def main():
            T.evaluate(block_m)

        return main

    def fake_compile(self, *args, **kwargs):
        compile_kwargs.append(dict(kwargs))
        return FakeKernel(f"plain-{kwargs['block_m']}")

    monkeypatch.setattr(JITImpl, "compile", fake_compile)

    first = plain_kernel(block_m=16)
    second = plain_kernel(block_m=32)

    assert plain_kernel.specialization is None
    assert first is not second
    assert len(plain_kernel._kernel_cache) == 2
    assert [kwargs["block_m"] for kwargs in compile_kwargs] == [16, 32]
    assert "specialization" not in plain_kernel._frontend_cache_key_data(((("block_m", 16),), None))
