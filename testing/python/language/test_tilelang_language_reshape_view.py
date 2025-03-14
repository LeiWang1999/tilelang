# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl


def reshape_test(N, M, dtype):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Buffer((N,), dtype),
            B: T.Buffer((N // M, M), dtype),
    ):
        with T.Kernel(1) as _:
            A_reshaped = T.reshape(A, [N // M, M])
            T.copy(A_reshaped, B)

    return main


def view_reshape_test(N, M, dtype):
    import tilelang.language as T

    @T.prim_func
    def main(
            A: T.Buffer((N,), dtype),
            B: T.Buffer((N // M, M), dtype),
    ):
        with T.Kernel(1) as _:
            A_viewed = T.view(A, shape=[N // M, M])
            T.copy(A_viewed, B)

    return main



def run_reshape(N, M, dtype):
    program = reshape_test(N, M, dtype)
    mod, params = tl.lower(program)
    profiler = tl.Profiler(mod, params, [1], tl.TensorSupplyType.Integer)

    def ref_program(A):
        import torch
        return A.reshape(N // M, M)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def run_view_reshape(N, M, dtype):
    program = view_reshape_test(N, M, dtype)
    mod, params = tl.lower(program)
    profiler = tl.Profiler(mod, params, [1], tl.TensorSupplyType.Integer)

    def ref_program(A):
        return A.view(N // M, M)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)



def test_reshape_view():
    # Test reshape
    run_reshape(1024, 32, "float32")
    run_reshape(2048, 64, "float16")
    
    # Test view reshape
    run_view_reshape(1024, 32, "float32")
    run_view_reshape(2048, 64, "float16")


if __name__ == "__main__":
    tilelang.testing.main() 