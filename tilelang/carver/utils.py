# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Optional, Union
from tvm import tir, IRModule
from tvm.tir import PrimFunc
from .arch import TileDevice
from .roller.policy import TensorCorePolicy, DefaultPolicy
from .roller.hint import Hint
from .matmul_analysis import get_tensorized_func_and_tags
import logging

logger = logging.getLogger(__name__)


def get_rasterization_code(pannel_width: int = 8) -> str:
    return f"""
        const int MAX_BLOCK_N = {pannel_width};
        const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
        const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
        const auto totalBlock = gridDim.x * gridDim.y;
        const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
        const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
        const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
        const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
        const auto bz = blockIdx.z;
        const dim3 blockIdx(bx, by, bz);
    """


def get_roller_hints_from_func(func_or_module: Union[tir.PrimFunc, IRModule],
                               arch: TileDevice,
                               topk: int = 10,
                               tensorcore_only: bool = False,
                               allow_gemv: bool = False) -> Optional[List[Hint]]:
    func = None
    if isinstance(func_or_module, tir.PrimFunc):
        func = func_or_module
    elif isinstance(func_or_module, IRModule):
        func = retrieve_func_from_module(func_or_module)
    else:
        raise ValueError("Not supported type: ", type(func_or_module))

    assert func is not None, "The function should not be None"

    if tensorcore_only:
        try:
            tensorized_func, tags = get_tensorized_func_and_tags(
                func, arch.target, allow_gemv=allow_gemv)
        except Exception as e_msg:
            logger.debug("Get tensorized func and tags failed: ", e_msg)
            tags = None
        if tags and tensorized_func:
            policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)
            return policy.emit_config(topk)
        else:
            return None
    else:
        policy = DefaultPolicy(func=func, arch=arch)
        tensorized_func = None
        try:
            tensorized_func, tags = get_tensorized_func_and_tags(
                func, arch.target, allow_gemv=allow_gemv)
        except Exception as e_msg:
            logger.debug("Get tensorized func and tags failed: ", e_msg)
            tags = None
        if tags and tensorized_func:
            policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)
        return policy.emit_config(topk)


def retrieve_func_from_module(ir_module: IRModule) -> PrimFunc:
    if not isinstance(ir_module, IRModule):
        raise ValueError("Not supported type: ", type(ir_module))
    assert len(ir_module.get_global_vars()) == 1, (
        "The optimized module should only have one global variable for default schedule.")
    func = list(ir_module.functions.values())[0]
    return func
