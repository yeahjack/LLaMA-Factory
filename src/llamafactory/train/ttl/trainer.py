# Copyright 2025 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Callable, Optional

from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.logging import get_logger
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)


class TTLTrainer(Seq2SeqTrainer):
    """Seq2Seq Trainer for Test-Time Learning (TTL) that allows injecting
    a custom loss function without altering the rest of Seq2SeqTrainer.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        compute_loss_func: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.finetuning_args = finetuning_args
        # External loss function: (outputs, inputs) -> Tensor
        self.compute_loss_func = compute_loss_func

    @override
    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(
                self.model, self.args, self.finetuning_args
            )
        return super().create_optimizer()

    @override
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs):
        if self.finetuning_args.disable_shuffling:
            import torch as _torch
            return _torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)
