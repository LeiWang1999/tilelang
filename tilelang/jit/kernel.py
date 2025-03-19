# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from typing import List, Union, Any, Callable, Literal, Optional, Dict
from tvm.target import Target
import tilelang
from tilelang import tvm as tvm
from tvm.tir import PrimFunc

from tilelang.jit.adapter import TorchDLPackKernelAdapter, BaseKernelAdapter, CtypesKernelAdapter, CythonKernelAdapter
from tilelang.utils.target import determine_target, AVALIABLE_TARGETS
from tilelang.profiler import Profiler, TensorSupplyType
from tilelang.engine.param import KernelParam, CompiledArtifact


class JITKernel(object):
    """
    A wrapper class for compiling and invoking TileLang (TVM TIR) functions as PyTorch-compatible functions.

    Attributes
    ----------
    artifact : CompiledArtifact
        The compiled artifact containing the runtime module and parameters.
    adapter : BaseKernelAdapter
        The adapter for the compiled function.
    torch_function : Callable
        The compiled function that can be invoked as a PyTorch-compatible function.
    """
    artifact: CompiledArtifact = None
    adapter: BaseKernelAdapter = None
    torch_function: Callable = None

    def __init__(
        self,
        func: PrimFunc = None,
        out_idx: Union[List[int], int] = None,
        execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
        verbose: bool = False,
        pass_configs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes a TorchFunction instance.

        Parameters
        ----------
        func : tvm.tir.PrimFunc, optional
            The TileLang TIR function to compile and wrap.
        out_idx : Union[List[int], int], optional
            Index(es) of the output tensors to return (default: None).
        execution_backend : Literal["dlpack", "ctypes"], optional
            Execution backend to use for kernel execution (default: "dlpack").
        target : Union[str, Target], optional
            Compilation target, either as a string or a TVM Target object (default: "auto").
        target_host : Union[str, Target], optional
            Target host for cross-compilation (default: None).
        verbose : bool, optional
            Whether to enable verbose output (default: False).
        pass_configs : dict, optional
            Additional keyword arguments to pass to the Compiler PassContext.
            Available options: 
                "tir.disable_vectorize": bool, default: False
                "tl.disable_tma_lower": bool, default: False
        """
        self.func = func
        self.out_idx = out_idx
        self.execution_backend = execution_backend
        self.target = target
        self.target_host = target_host
        self.verbose = verbose

        if pass_configs is None:
            pass_configs = {}
        self.pass_configs = pass_configs

        # If the target is specified as a string, validate it and convert it to a TVM Target.
        if isinstance(target, str):
            assert target in AVALIABLE_TARGETS, f"Invalid target: {target}"
            target = determine_target(target)

        # Ensure the target is always a TVM Target object.
        target = Target(target)

        # Validate the execution backend.
        assert execution_backend in ["dlpack", "ctypes",
                                     "cython"], f"Invalid execution backend. {execution_backend}"
        if execution_backend == "cython":
            from tilelang.contrib.cc import get_cplus_compiler
            assert get_cplus_compiler(
            ) is not None, "Cython backend requires a C++ compiler, please install or use other backends."

        # Compile the TileLang function and create a kernel adapter for execution.
        adapter = self._compile_and_create_adapter(func)

        # The adapter's function is assigned as the callable function for this instance.
        self.adapter = adapter
        self.torch_function = adapter.func

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Invokes the compiled function with the given arguments.

        Parameters
        ----------
        *args : Any
            Positional arguments for the function.
        **kwds : Any
            Keyword arguments for the function.

        Returns
        -------
        Any
            The result of the function execution.
        """
        return self.torch_function(*args, **kwds)

    def _compile_and_create_adapter(self, tilelang_func: PrimFunc) -> BaseKernelAdapter:
        """
        Compiles the given TileLang PrimFunc using TVM and creates a kernel adapter.

        Parameters
        ----------
        tilelang_func : tvm.tir.PrimFunc
            The TileLang (TVM TIR) function to compile.

        Returns
        -------
        BaseKernelAdapter
            The compiled and ready-to-run kernel adapter.
        """
        verbose = self.verbose
        target = self.target
        target_host = self.target_host
        out_idx = self.out_idx
        execution_backend = self.execution_backend
        pass_configs = self.pass_configs

        # Compile the function with TVM, optimizing with shared memory lowering.
        with tvm.transform.PassContext(opt_level=3, config=pass_configs):
            artifact = tilelang.lower(tilelang_func, target=target, target_host=target_host)

        self.artifact = artifact

        # Create an adapter based on the specified execution backend.
        if execution_backend == "dlpack":
            # Use TorchDLPackKernelAdapter for interoperability with PyTorch via DLPack.
            # But we need to ensure that the runtime is enabled and the runtime module is not None.
            assert tvm.runtime.enabled("llvm"), "DLPack backend requires LLVM runtime."
            assert artifact.rt_mod is not None, "DLPack backend requires a runtime module."
            adapter = TorchDLPackKernelAdapter(
                artifact.rt_mod, params=artifact.params, result_idx=out_idx)
        elif execution_backend == "ctypes":
            adapter = CtypesKernelAdapter(
                params=artifact.params,
                result_idx=out_idx,
                target=target,
                func_or_mod=tilelang_func,
                kernel_global_source=artifact.kernel_source,
                verbose=verbose,
                pass_configs=pass_configs,
            )
        elif execution_backend == "cython":
            adapter = CythonKernelAdapter(
                params=artifact.params,
                result_idx=out_idx,
                target=target,
                func_or_mod=tilelang_func,
                kernel_global_source=artifact.kernel_source,
                verbose=verbose,
                pass_configs=pass_configs,
            )
        else:
            # Handle invalid backend.
            raise ValueError(f"Invalid execution backend: {execution_backend}")

        return adapter

    @classmethod
    def from_tilelang_function(cls, tilelang_func: PrimFunc, **kwargs):
        """
        Alternative constructor to create a TorchFunction directly from a TileLang PrimFunc.

        Parameters
        ----------
        tilelang_func : tvm.tir.PrimFunc
            The TileLang (TVM TIR) function to compile.
        **kwargs : dict
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        TorchFunction
            An instance of TorchFunction wrapping the compiled function.
        """
        return cls(func=tilelang_func, **kwargs)

    def get_profiler(self,
                     tensor_supply_type: TensorSupplyType = TensorSupplyType.Auto) -> Profiler:
        """
        Creates a profiler to benchmark the compiled runtime module.

        Parameters
        ----------
        tensor_supply_type : TensorSupplyType, optional
            The type of input tensors to supply for profiling (default: TensorSupplyType.Auto).

        Returns
        -------
        Profiler
            A Profiler instance for benchmarking the runtime module.
        """
        return Profiler(self.params, self.out_idx,
                        tensor_supply_type).with_default_adapter(self.adapter)

    def get_kernel_source(self) -> str:
        """
        Returns the source code of the compiled kernel function.

        Returns
        -------
        str
            The source code of the compiled kernel function.
        """
        if self.execution_backend in {"ctypes", "cython"}:
            return self.adapter.get_kernel_source()
        return self.artifact.kernel_source

    def get_host_source(self) -> str:
        """
        Returns the source code of the host function.
        """
        return str(self.artifact.host_mod)

    def run_once(self, func: Optional[Callable] = None) -> None:
        return self.get_profiler().run_once(func)

    @property
    def params(self) -> List[KernelParam]:
        return self.artifact.params

    @property
    def kernel_source(self) -> str:
        return self.artifact.kernel_source

    @property
    def host_source(self) -> str:
        return str(self.artifact.host_mod)
