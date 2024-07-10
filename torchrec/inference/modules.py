#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, cast, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.quantization as quant
import torchrec as trec
import torchrec.distributed as trec_dist
import torchrec.quant as trec_quant
from torch.fx.passes.split_utils import getattr_recursive
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fused_params import (
    FUSED_PARAM_BOUNDS_CHECK_MODE,
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    FUSED_PARAM_REGISTER_TBE_BOOL,
)
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.planner.storage_reservations import (
    FixedPercentageStorageReservation,
)
from torchrec.distributed.quant_embedding import QuantEmbeddingCollectionSharder
from torchrec.distributed.quant_embeddingbag import (
    QuantEmbeddingBagCollectionSharder,
    QuantFeatureProcessedEmbeddingBagCollectionSharder,
)
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.types import (
    BoundsCheckMode,
    ModuleSharder,
    ShardingPlan,
    ShardingType,
)

from torchrec.modules.embedding_configs import QuantConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
    EmbeddingCollection,
    EmbeddingCollectionInterface,
)
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection

from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
    EmbeddingCollection as QuantEmbeddingCollection,
    FeatureProcessedEmbeddingBagCollection as QuantFeatureProcessedEmbeddingBagCollection,
    quant_prep_enable_register_tbes,
)

logger: logging.Logger = logging.getLogger(__name__)


def trim_torch_package_prefix_from_typename(typename: str) -> str:
    if typename.startswith("<torch_package_"):
        # Trim off <torch_package_x> prefix.
        typename = ".".join(typename.split(".")[1:])
    return typename


DEFAULT_FUSED_PARAMS: Dict[str, Any] = {
    FUSED_PARAM_REGISTER_TBE_BOOL: True,
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS: True,
    FUSED_PARAM_BOUNDS_CHECK_MODE: BoundsCheckMode.NONE,
}

DEFAULT_SHARDERS: List[ModuleSharder[torch.nn.Module]] = [
    cast(
        ModuleSharder[torch.nn.Module],
        QuantEmbeddingBagCollectionSharder(fused_params=DEFAULT_FUSED_PARAMS),
    ),
    cast(
        ModuleSharder[torch.nn.Module],
        QuantEmbeddingCollectionSharder(fused_params=DEFAULT_FUSED_PARAMS),
    ),
    cast(
        ModuleSharder[torch.nn.Module],
        QuantFeatureProcessedEmbeddingBagCollectionSharder(
            fused_params=DEFAULT_FUSED_PARAMS
        ),
    ),
]

DEFAULT_QUANT_MAPPING: Dict[str, Type[torch.nn.Module]] = {
    trim_torch_package_prefix_from_typename(
        torch.typename(EmbeddingBagCollection)
    ): QuantEmbeddingBagCollection,
    trim_torch_package_prefix_from_typename(
        torch.typename(EmbeddingCollection)
    ): QuantEmbeddingCollection,
}

DEFAULT_QUANTIZATION_DTYPE: torch.dtype = torch.int8

FEATURE_PROCESSED_EBC_TYPE: str = trim_torch_package_prefix_from_typename(
    torch.typename(FeatureProcessedEmbeddingBagCollection)
)


def quantize_feature(
    module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, ...]:
    return tuple(
        [
            (
                input.half()
                if isinstance(input, torch.Tensor)
                and input.dtype in [torch.float32, torch.float64]
                else input
            )
            for input in inputs
        ]
    )


def quantize_embeddings(
    module: nn.Module,
    dtype: torch.dtype,
    inplace: bool,
    additional_qconfig_spec_keys: Optional[List[Type[nn.Module]]] = None,
    additional_mapping: Optional[Dict[Type[nn.Module], Type[nn.Module]]] = None,
    output_dtype: torch.dtype = torch.float,
    per_table_weight_dtype: Optional[Dict[str, torch.dtype]] = None,
) -> nn.Module:
    qconfig = QuantConfig(
        activation=quant.PlaceholderObserver.with_args(dtype=output_dtype),
        weight=quant.PlaceholderObserver.with_args(dtype=dtype),
        per_table_weight_dtype=per_table_weight_dtype,
    )
    qconfig_spec: Dict[Type[nn.Module], QuantConfig] = {
        trec.EmbeddingBagCollection: qconfig,
    }
    mapping: Dict[Type[nn.Module], Type[nn.Module]] = {
        trec.EmbeddingBagCollection: trec_quant.EmbeddingBagCollection,
    }
    if additional_qconfig_spec_keys is not None:
        for t in additional_qconfig_spec_keys:
            qconfig_spec[t] = qconfig
    if additional_mapping is not None:
        mapping.update(additional_mapping)
    return quant.quantize_dynamic(
        module,
        qconfig_spec=qconfig_spec,
        mapping=mapping,
        inplace=inplace,
    )


@dataclass
class QualNameMetadata:
    need_preproc: bool


@dataclass
class BatchingMetadata:
    """
    Metadata class for batching, this should be kept in sync with the C++ definition.
    """

    type: str
    # cpu or cuda
    device: str
    # list of tensor suffixes to deserialize to pinned memory (e.g. "lengths")
    # use "" (empty string) to pin without suffix
    pinned: List[str]


class PredictFactory(abc.ABC):
    """
    Creates a model (with already learned weights) to be used inference time.
    """

    @abc.abstractmethod
    def create_predict_module(self) -> nn.Module:
        """
        Returns already sharded model with allocated weights.
        state_dict() must match TransformModule.transform_state_dict().
        It assumes that torch.distributed.init_process_group was already called
        and will shard model according to torch.distributed.get_world_size().
        """
        pass

    @abc.abstractmethod
    def batching_metadata(self) -> Dict[str, BatchingMetadata]:
        """
        Returns a dict from input name to BatchingMetadata. This infomation is used for batching for input requests.
        """
        pass

    def batching_metadata_json(self) -> str:
        """
        Serialize the batching metadata to JSON, for ease of parsing with torch::deploy environments.
        """
        return json.dumps(
            {key: asdict(value) for key, value in self.batching_metadata().items()}
        )

    @abc.abstractmethod
    def result_metadata(self) -> str:
        """
        Returns a string which represents the result type. This information is used for result split.
        """
        pass

    @abc.abstractmethod
    def run_weights_independent_tranformations(
        self, predict_module: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Run transformations that don't rely on weights of the predict module. e.g. fx tracing, model
        split etc.
        """
        pass

    @abc.abstractmethod
    def run_weights_dependent_transformations(
        self, predict_module: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Run transformations that depends on weights of the predict module. e.g. lowering to a backend.
        """
        pass

    def qualname_metadata(self) -> Dict[str, QualNameMetadata]:
        """
        Returns a dict from qualname (method name) to QualNameMetadata. This is additional information for execution of specific methods of the model.
        """
        return {}

    def qualname_metadata_json(self) -> str:
        """
        Serialize the qualname metadata to JSON, for ease of parsing with torch::deploy environments.
        """
        return json.dumps(
            {key: asdict(value) for key, value in self.qualname_metadata().items()}
        )

    def model_inputs_data(self) -> Dict[str, Any]:
        """
        Returns a dict of various data for benchmarking input generation.
        """
        return {}


class PredictModule(nn.Module):
    """
    Interface for modules to work in a torch.deploy based backend. Users should
    override predict_forward to convert batch input format to module input format.

    Call Args:
        batch: a dict of input tensors

    Returns:
        output: a dict of output tensors

    Args:
        module: the actual predict module
        device: the primary device for this module that will be used in forward calls.

    Example::

        module = PredictModule(torch.device("cuda", torch.cuda.current_device()))
    """

    def __init__(
        self,
        module: nn.Module,
    ) -> None:
        super().__init__()
        self._module: nn.Module = module
        # lazy init device from thread inited device guard
        self._device: Optional[torch.device] = None
        self._module.eval()

    @property
    def predict_module(
        self,
    ) -> nn.Module:
        return self._module

    @abc.abstractmethod
    # pyre-fixme[3]
    def predict_forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        pass

    # pyre-fixme[3]
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        if self._device is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        with torch.cuda.device(self._device), torch.inference_mode():
            return self.predict_forward(batch)

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        # pyre-fixme[19]: Expected 0 positional arguments.
        return self._module.state_dict(destination, prefix, keep_vars)


def quantize_dense(
    predict_module: PredictModule,
    dtype: torch.dtype,
    additional_embedding_module_type: List[Type[nn.Module]] = [],
) -> nn.Module:
    module = predict_module.predict_module
    reassign = {}

    for name, mod in module.named_children():
        # both fused modules and observed custom modules are
        # swapped as one unit
        if not (
            isinstance(mod, EmbeddingBagCollectionInterface)
            or isinstance(mod, EmbeddingCollectionInterface)
            or any([type(mod) is clazz for clazz in additional_embedding_module_type])
        ):
            if dtype == torch.half:
                new_mod = mod.half()
                new_mod.register_forward_pre_hook(quantize_feature)
                reassign[name] = new_mod
            else:
                raise NotImplementedError(
                    "only fp16 is supported for non-embedding module lowering"
                )
    for key, value in reassign.items():
        module._modules[key] = value
    return predict_module


def quantize_inference_model(
    model: torch.nn.Module,
    quantization_mapping: Optional[Dict[str, Type[torch.nn.Module]]] = None,
    per_table_weight_dtype: Optional[Dict[str, torch.dtype]] = None,
    fp_weight_dtype: torch.dtype = DEFAULT_QUANTIZATION_DTYPE,
) -> torch.nn.Module:
    """
    Quantize the model.
    """

    if quantization_mapping is None:
        quantization_mapping = DEFAULT_QUANT_MAPPING

    def _quantize_fp_module(
        model: torch.nn.Module,
        fp_module: FeatureProcessedEmbeddingBagCollection,
        fp_module_fqn: str,
        activation_dtype: torch.dtype = torch.float,
        weight_dtype: torch.dtype = DEFAULT_QUANTIZATION_DTYPE,
    ) -> None:
        """
        If FeatureProcessedEmbeddingBagCollection is found, quantize via direct module swap.
        """
        fp_module.qconfig = quant.QConfig(
            activation=quant.PlaceholderObserver.with_args(dtype=activation_dtype),
            weight=quant.PlaceholderObserver.with_args(dtype=weight_dtype),
        )
        # ie. "root.submodule.feature_processed_mod" -> "root.submodule", "feature_processed_mod"
        fp_ebc_parent_fqn, fp_ebc_name = fp_module_fqn.rsplit(".", 1)
        fp_ebc_parent = getattr_recursive(model, fp_ebc_parent_fqn)
        fp_ebc_parent.register_module(
            fp_ebc_name,
            QuantFeatureProcessedEmbeddingBagCollection.from_float(fp_module),
        )

    additional_qconfig_spec_keys = []
    additional_mapping = {}

    for n, m in model.named_modules():
        typename = trim_torch_package_prefix_from_typename(torch.typename(m))

        if typename in quantization_mapping:
            additional_qconfig_spec_keys.append(type(m))
            additional_mapping[type(m)] = quantization_mapping[typename]
        elif typename == FEATURE_PROCESSED_EBC_TYPE:
            # handle the fp ebc separately
            _quantize_fp_module(model, m, n, weight_dtype=fp_weight_dtype)

    quant_prep_enable_register_tbes(model, list(additional_mapping.keys()))
    quantize_embeddings(
        model,
        dtype=DEFAULT_QUANTIZATION_DTYPE,
        additional_qconfig_spec_keys=additional_qconfig_spec_keys,
        additional_mapping=additional_mapping,
        inplace=True,
        per_table_weight_dtype=per_table_weight_dtype,
    )

    logger.info(
        f"Default quantization dtype is {DEFAULT_QUANTIZATION_DTYPE}, {per_table_weight_dtype=}."
    )

    return model


def shard_quant_model(
    model: torch.nn.Module,
    world_size: int = 1,
    compute_device: str = "cuda",
    sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
    fused_params: Optional[Dict[str, Any]] = None,
    device_memory_size: Optional[int] = None,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
) -> Tuple[torch.nn.Module, ShardingPlan]:
    """
    Shard the model.
    """

    if constraints is None:
        table_fqns = []
        for name, _ in model.named_modules():
            if "table" in name:
                table_fqns.append(name.split(".")[-1])

        # Default table wise constraints
        constraints = {}
        for name in table_fqns:
            constraints[name] = ParameterConstraints(
                sharding_types=[ShardingType.TABLE_WISE.value],
                compute_kernels=[EmbeddingComputeKernel.QUANT.value],
            )

    if device_memory_size is not None:
        hbm_cap = device_memory_size
    elif torch.cuda.is_available() and compute_device == "cuda":
        hbm_cap = torch.cuda.get_device_properties(
            f"cuda:{torch.cuda.current_device()}"
        ).total_memory
    else:
        hbm_cap = None

    topology = trec_dist.planner.Topology(
        world_size=world_size,
        compute_device=compute_device,
        local_world_size=world_size,
        hbm_cap=hbm_cap,
    )
    batch_size = 1
    model_plan = trec_dist.planner.EmbeddingShardingPlanner(
        topology=topology,
        batch_size=batch_size,
        constraints=constraints,
        enumerator=EmbeddingEnumerator(
            topology=topology,
            batch_size=batch_size,
            constraints=constraints,
            estimator=[
                EmbeddingPerfEstimator(
                    topology=topology, constraints=constraints, is_inference=True
                ),
                EmbeddingStorageEstimator(topology=topology, constraints=constraints),
            ],
        ),
        storage_reservation=FixedPercentageStorageReservation(
            percentage=0.0,
        ),
    ).plan(
        model,
        sharders if sharders else DEFAULT_SHARDERS,
    )

    model = _shard_modules(
        module=model,
        device=torch.device("meta"),
        plan=model_plan,
        env=trec_dist.ShardingEnv.from_local(
            world_size,
            0,
        ),
        sharders=sharders if sharders else DEFAULT_SHARDERS,
    )

    return model, model_plan
