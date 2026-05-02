#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec import optim as trec_optim
from torchrec.distributed.utils import (
    _EMB_OPT_TYPE_TO_OPTIMIZER_CLASS,
    _OPTIMIZER_CLASS_TO_EMB_OPT_TYPE,
    optimizer_type_to_emb_opt_type,
)
from torchrec.modules.fused_embedding_modules import convert_optimizer_type_and_kwargs


class AdaDeltaRMSPropMappingTest(unittest.TestCase):
    def test_class_to_emb_opt_type(self) -> None:
        self.assertEqual(
            _OPTIMIZER_CLASS_TO_EMB_OPT_TYPE[trec_optim.AdaDelta],
            EmbOptimType.ADADELTA,
        )
        self.assertEqual(
            _OPTIMIZER_CLASS_TO_EMB_OPT_TYPE[trec_optim.RMSProp],
            EmbOptimType.RMSPROP,
        )

    def test_emb_opt_type_to_class(self) -> None:
        self.assertIs(
            _EMB_OPT_TYPE_TO_OPTIMIZER_CLASS[EmbOptimType.ADADELTA],
            trec_optim.AdaDelta,
        )
        self.assertIs(
            _EMB_OPT_TYPE_TO_OPTIMIZER_CLASS[EmbOptimType.RMSPROP],
            trec_optim.RMSProp,
        )

    def test_optimizer_type_to_emb_opt_type(self) -> None:
        self.assertEqual(
            optimizer_type_to_emb_opt_type(trec_optim.AdaDelta),
            EmbOptimType.ADADELTA,
        )
        self.assertEqual(
            optimizer_type_to_emb_opt_type(trec_optim.RMSProp),
            EmbOptimType.RMSPROP,
        )

    def test_convert_adadelta_kwargs_aliases_rho_to_beta1(self) -> None:
        result = convert_optimizer_type_and_kwargs(
            trec_optim.AdaDelta,
            {"lr": 0.01, "rho": 0.9, "eps": 1e-6, "weight_decay": 0.0},
        )
        assert result is not None
        opt_type, kwargs = result
        self.assertEqual(opt_type, EmbOptimType.ADADELTA)
        self.assertEqual(kwargs["learning_rate"], 0.01)
        self.assertEqual(kwargs["beta1"], 0.9)
        self.assertEqual(kwargs["eps"], 1e-6)
        self.assertEqual(kwargs["weight_decay"], 0.0)
        self.assertNotIn("rho", kwargs)
        self.assertNotIn("lr", kwargs)

    def test_convert_rmsprop_kwargs_aliases_alpha_to_beta1(self) -> None:
        result = convert_optimizer_type_and_kwargs(
            trec_optim.RMSProp,
            {"lr": 0.001, "alpha": 0.99, "eps": 1e-8, "weight_decay": 0.0},
        )
        assert result is not None
        opt_type, kwargs = result
        self.assertEqual(opt_type, EmbOptimType.RMSPROP)
        self.assertEqual(kwargs["learning_rate"], 0.001)
        self.assertEqual(kwargs["beta1"], 0.99)
        self.assertEqual(kwargs["eps"], 1e-8)
        self.assertNotIn("alpha", kwargs)
        self.assertNotIn("lr", kwargs)


if __name__ == "__main__":
    unittest.main()
