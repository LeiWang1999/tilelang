# Shape-Guided Variant Specialization

TileLang JIT kernels can use runtime tensor metadata to choose static compile-time values before a kernel variant is compiled. This lets one Python kernel family lazily produce a small set of reusable static variants.

```python
import tilelang
import tilelang.language as T

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
```

For this specialization, calls with `A.shape[0] == 1` and `A.shape[0] == 2` both select `block_m=16` and reuse the same JIT variant. Calls with `A.shape[0] == 33` select `block_m=64` and compile or reuse a separate variant.

The selected values are injected as Python compile-time keyword arguments before PrimFunc generation. They are not appended as extra runtime kernel parameters.

## API

`tilelang.arg(name)` creates a metadata reference to a runtime argument. The MVP supports shape indexing through `tilelang.arg("A").shape[i]`.

`tilelang.bucket(expr, buckets, policy="ceil", overflow="error")` maps an integer metadata expression to the first bucket that is greater than or equal to the runtime value. Buckets must be non-empty, integer-valued, and strictly increasing.

`tilelang.specialize(..., require=None, compile="on_miss")` attaches a specialization spec to a JIT function. Static field values such as `block_n=128` are preserved in the selected variant config. Selector fields such as `block_m=tilelang.bucket(...)` are evaluated at call time.

`require=[...]` accepts metadata comparison expressions such as `m > 0` or `tilelang.arg("A").shape[1] == tilelang.arg("B").shape[1]`. Requirements are evaluated on the Python side before compile or launch.

## MVP Scope

Only `compile="on_miss"` is implemented. Ahead-of-time modes such as `compile="ahead"` and `compile="ahead_host_dispatch"` are future work and are rejected explicitly.

Arbitrary Python callables or lambdas are not accepted as specialization selectors. Selectors must be built from the TileLang metadata expression API so the selected variant config can be normalized into deterministic cache keys.

Shape-guided variant specialization is not a generic runtime-parameterized kernel mode. Values selected by `tilelang.specialize` are compile-time values used during PrimFunc generation.
