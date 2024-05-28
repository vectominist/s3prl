# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This source code contains parts from
#
#     wav2vec/wav2vec_model.py
#
# with the following copyright statement.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates.
#

from omegaconf import DictConfig
from torch import nn

from .nemo import Serialization


class ST2VecEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.wav2spec = Serialization.from_config_dict(cfg.preprocessor)
        self.feature_encoder = Serialization.from_config_dict(cfg.feature_encoder)

    def forward(
        self,
        wavs,
        wav_lens,
    ) -> tuple:
        # specs: [B, C, T]
        specs, specs_len = self.wav2spec(input_signal=wavs, length=wav_lens)
        features, feature_lens, _ = self.feature_encoder(specs, specs_len)
        # [B, D, T] => [B, T, D]
        features = features.transpose(1, 2)
        return features, feature_lens
