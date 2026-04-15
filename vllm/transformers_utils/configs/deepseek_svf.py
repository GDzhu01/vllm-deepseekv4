# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig


class DeepseekSVFConfig(PretrainedConfig):
    model_type = "deepseek_svf"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
