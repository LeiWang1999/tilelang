import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm import DataType
from tvm import tir
import torch
from utils import torch_convert_bit_twiddling, torch_convert


def _tir_u8_to_f4_to_bf16(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, scale: tir.PrimExpr,
                          dtype: str):
    assert nbit == 4
    assert dtype == "bfloat16"
    assert val.dtype == "uint8"
    mask = tir.const((1 << nbit) - 1, "uint16")
    f4 = (val >> (pos.astype("uint16") * tir.const(nbit, "uint16"))) & mask
    s = f4 >> tir.const(3, "uint16")
    e_f4 = (f4 & tir.const(6, "uint16")) >> tir.const(1, "uint16")
    # Exponential bias between f4 and bf16 is 2^(8-1) - 2^(2-1) = 126
    e_bf16 = e_f4 + tir.const(126, "uint16")
    # Scale is the exponential part, within the representation of uint8
    # To handle the overflow, we use the max function to limit the exponential part to 8 bits
    e_bf16 = T.min(e_bf16 + scale, tir.const((1 << 8) - 1, "uint16"))
    m_f4 = f4 & tir.const(1, "uint16")
    val_bf16 = tir.reinterpret("bfloat16",
                               ((((s << tir.const(8, "uint16")) | e_bf16) << tir.const(7, "uint16"))
                                | (m_f4 << tir.const(6, "uint16"))).astype("uint16"))
    return val_bf16


def get_configs():
    import itertools
    iter_params = dict(
        block_M=[64, 128, 256],
        block_N=[64, 128, 256],
        block_K=[64, 128, 256],
        num_stages=[0, 2],
        threads=[128, 256, 512],
        split=[1, 2],
    )
    return [{
        k: v for k, v in zip(iter_params, values)
    } for values in itertools.product(*iter_params.values())]


@tilelang.autotune(configs=get_configs(),)
@tilelang.jit(out_idx=[-1],)
def matmul(M,
           N,
           K,
           in_dtype,
           out_dtype,
           accum_dtype,
           source_format='uint',
           num_bits=4,
           scale_size=32,
           fast_dequant=True,
           block_M=256,
           block_N=128,
           block_K=128,
           num_stages=2,
           threads=256,
           split=1):

    num_elems_per_byte = 8 // num_bits
    storage_dtype = "uint8"
    QK = K // num_elems_per_byte
    Block_QK = block_K // num_elems_per_byte
    A_shape = (M, K)
    B_shape = (N, QK)
    Scale_shape = (N, K // scale_size)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, Block_QK)
    B_dequantize_shared_shape = (block_N, block_K)
    assert K % (block_K * split) == 0

    from tilelang.quantize import get_mxfp_intrin_group
    # fast_dequant_bf16_fp4_twiddling
    mxfp_intrin_info = get_mxfp_intrin_group(
        out_dtype=in_dtype,
        source_format=source_format,
        source_bit=num_bits,
        storage_dtype=storage_dtype,
        use_twiddling=True,
    )
    import_source = mxfp_intrin_info["c_source"]
    func_name = mxfp_intrin_info["func_name"]
    assert import_source is not None, "mxfp_intrin_info is not found"
    assert func_name is not None, "mxfp_intrin_info is not found"
    import_source = import_source

    def get_fast_dequant_twiddling_func(in_dtype="fp4", out_dtype="bfloat16"):
        assert in_dtype in ["fp4"]
        assert out_dtype in ["bfloat16"]

        # Some variables for dequantization in each thread
        MAX_TRANSACTION_SIZE_BITS = 128
        local_size = MAX_TRANSACTION_SIZE_BITS // DataType(out_dtype).bits
        local_compress_size = local_size // num_elems_per_byte

        @T.macro
        def fast_dequant_bf16_fp4_twiddling(B_shared, B_dequantize_shared, Scale, k):
            # import fast_dequantize plugin
            T.import_source(import_source)

            tx = T.get_thread_binding()
            bx = T.get_block_binding(0)

            B_local_thread = T.alloc_local((local_compress_size,), storage_dtype)
            B_dequantize_local_thread = T.alloc_local((local_size,), out_dtype)
            Scale_local_thread = T.alloc_local((1,), storage_dtype)
            Scale_local_thread_exponent = T.alloc_local((1,), "float32")

            for i in T.serial(0, block_N * block_K // threads // local_size):
                # First, load data from share memory to register.
                # Prepare for dequant.
                index_base = i * threads * local_compress_size + tx * local_compress_size
                for v in T.vectorized(0, local_compress_size):
                    index = index_base + v
                    B_local_thread[v] = B_shared[index // Block_QK, index % Block_QK]
                index_scale = index_base // (scale_size // num_elems_per_byte)
                si = index_scale // (block_K // scale_size)
                sj = index_scale % (block_K // scale_size)
                Scale_local_thread[0] = Scale[bx * block_N + si, k * block_K // scale_size + sj]
                Scale_local_thread_exponent[0] = T.exp2(
                    T.cast(Scale_local_thread[0] - 127, "float"))

                # Then, dequant.
                T.call_extern(
                    func_name,
                    T.address_of(B_local_thread[0]),
                    T.address_of(B_dequantize_local_thread[0]),
                    1,
                    dtype=out_dtype,
                )

                # Finally, store the dequantized data to shared memory.
                for v in T.Parallel(local_size):
                    B_dequantize_local_thread[v] *= Scale_local_thread_exponent[0]

                for v in T.vectorized(0, local_size):
                    index = i * threads * local_size + tx * local_size + v
                    B_dequantize_shared[index // block_K,
                                        index % block_K] = B_dequantize_local_thread[v]

        return fast_dequant_bf16_fp4_twiddling

    def get_simple_dequant_func(in_dtype="fp4", out_dtype="bfloat16"):
        assert in_dtype in ["fp4"]
        assert out_dtype in ["bfloat16"]

        @T.macro
        def simple_dequant_bf16_fp4(B_shared, B_dequantize_shared, Scale, k):
            B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
            B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)

            bx = T.get_block_binding(0)
            T.copy(B_shared, B_local)
            for i, j in T.Parallel(block_N, block_K):
                B_dequantize_local[i, j] = _tir_u8_to_f4_to_bf16(
                    num_bits,
                    B_local[i, j // num_elems_per_byte],
                    j % num_elems_per_byte,
                    Scale[
                        bx * block_N + i, k * block_K // scale_size + j //
                        scale_size],  # Scale is the exponential part, within the representation of uint8
                    dtype=in_dtype,
                )
            T.copy(B_dequantize_local, B_dequantize_shared)

        return simple_dequant_bf16_fp4

    @T.prim_func
    def main(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, storage_dtype),
            Scale: T.Tensor(Scale_shape, storage_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
            B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)

            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            T.annotate_layout({
                C_shared: tilelang.layout.make_swizzled_layout(C_shared),
            })

            T.clear(C_local)
            for k in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                if fast_dequant:
                    get_fast_dequant_twiddling_func()(B_shared, B_dequantize_shared, Scale, k)
                else:
                    get_simple_dequant_func()(B_shared, B_dequantize_shared, Scale, k)
                T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M:(by + 1) * block_M, bx * block_N:(bx + 1) * block_N])

    return main


def ref_program_twiddling(A, qB, Scale):
    dtypeC = "bfloat16"
    B = torch_convert_bit_twiddling(qB)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = B[i][j] * (2**(Scale[i][j // 32] - 127))
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def ref_program_simple(A, qB, Scale):
    dtypeC = "bfloat16"
    B = torch_convert(qB)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = B[i][j] * (2**(Scale[i][j // 32] - 127))
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def main(m=256, n=256, k=256, scale_size=32, fast_dequant=True, tune=False):
    total_flops = 2 * m * n * k

    if tune:
        kernel = matmul(
            m, n, k, "bfloat16", "bfloat16", "float32", num_bits=4, scale_size=scale_size)
    else:
        kernel = matmul(
            m,
            n,
            k,
            "bfloat16",
            "bfloat16",
            "float32",
            num_bits=4,
            scale_size=scale_size,
            block_M=256,
            block_N=128,
            block_K=128,
            num_stages=2,
            threads=256,
            split=1)

    profiler = kernel.get_profiler(tilelang.TensorSupplyType.Auto)

    if fast_dequant:
        profiler.assert_allclose(ref_program_twiddling, rtol=0.01, atol=0.01)
    else:
        profiler.assert_allclose(ref_program_simple, rtol=0.01, atol=0.01)
    print("All checks pass.")
    latency = profiler.do_bench(warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))
    print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))


if __name__ == "__main__":
    M, N, K = 256, 256, 256
    scale_size = 32
    main(M, N, K, scale_size, fast_dequant=True)
    main(M, N, K, scale_size, fast_dequant=False)
