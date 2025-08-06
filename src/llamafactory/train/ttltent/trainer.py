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

import math
from typing import TYPE_CHECKING

import torch
from torch.nn import CrossEntropyLoss
from transformers import Seq2SeqTrainer
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)


# Loss Balancing Classes
class DynamicWeightBalancer:
    def __init__(self, alpha=0.12, initial_weights=[1.0, 1.0, 1.0]):
        self.alpha = alpha
        self.weights = torch.tensor(initial_weights, dtype=torch.float32)
        self.loss_history = []

    def update_weights(self, losses):
        """根据loss的相对大小动态调整权重"""
        losses = torch.tensor(losses)
        if len(self.loss_history) > 0:
            prev_losses = torch.tensor(self.loss_history[-1])
            loss_ratios = losses / (prev_losses + 1e-8)
            self.weights = self.weights * (2.0 - loss_ratios * self.alpha)
            self.weights = torch.clamp(self.weights, 0.1, 10.0)

        self.loss_history.append(losses.tolist())
        return self.weights


class GradientMagnitudeBalancer:
    def __init__(self, target_ratio=1.0, momentum=0.9):
        self.target_ratio = target_ratio
        self.momentum = momentum
        self.ema_grad_norms = None

    def compute_balanced_weights(self, model, losses):
        """基于梯度大小计算平衡权重"""
        grad_norms = []

        # 获取可训练参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        for loss in losses:
            if loss.requires_grad and loss.item() > 0:
                try:
                    grads = torch.autograd.grad(
                        loss, trainable_params, retain_graph=True, create_graph=False, allow_unused=True
                    )
                    # 过滤None梯度
                    valid_grads = [g for g in grads if g is not None]
                    if valid_grads:
                        grad_norm = torch.sqrt(sum(torch.sum(g**2) for g in valid_grads))
                        grad_norms.append(grad_norm)
                    else:
                        grad_norms.append(torch.tensor(1.0, device=loss.device))
                except RuntimeError:
                    # 如果梯度计算失败，使用默认值
                    grad_norms.append(torch.tensor(1.0, device=loss.device))
            else:
                grad_norms.append(torch.tensor(1.0, device=loss.device))

        grad_norms = torch.stack(grad_norms)

        if self.ema_grad_norms is None:
            self.ema_grad_norms = grad_norms
        else:
            self.ema_grad_norms = self.momentum * self.ema_grad_norms + (1 - self.momentum) * grad_norms

        avg_norm = self.ema_grad_norms.mean()
        weights = avg_norm / (self.ema_grad_norms + 1e-8)

        return weights


class UncertaintyWeightBalancer(torch.nn.Module):
    def __init__(self, num_tasks=3):
        super().__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """基于学习的不确定性计算权重"""
        losses = torch.stack(losses)
        precision = torch.exp(-self.log_vars)
        weighted_losses = precision * losses + self.log_vars
        return weighted_losses.sum(), precision


class MovingAverageBalancer:
    def __init__(self, window_size=100, target_ratio=1.0):
        self.window_size = window_size
        self.target_ratio = target_ratio
        self.loss_history = {"ttl": [], "tent": [], "kl": []}

    def update_weights(self, losses, loss_names=["ttl", "tent", "kl"]):
        """基于移动平均调整权重"""
        for i, name in enumerate(loss_names):
            if i < len(losses):
                self.loss_history[name].append(losses[i].item())
                if len(self.loss_history[name]) > self.window_size:
                    self.loss_history[name] = self.loss_history[name][-self.window_size :]

        if len(self.loss_history["ttl"]) < 10:
            return torch.tensor([1.0, 1.0, 1.0])

        # 计算移动平均
        avg_losses = []
        for name in loss_names:
            if self.loss_history[name]:
                avg_loss = sum(self.loss_history[name]) / len(self.loss_history[name])
                avg_losses.append(avg_loss)
            else:
                avg_losses.append(1.0)

        # 归一化权重
        max_avg = max(avg_losses)
        weights = [max_avg / (avg + 1e-8) for avg in avg_losses]

        return torch.tensor(weights)


class AdaptiveLossScaler:
    def __init__(self, initial_scale=1.0, scale_factor=1.1, patience=10):
        self.scales = torch.tensor([initial_scale, initial_scale, initial_scale])
        self.scale_factor = scale_factor
        self.patience = patience
        self.loss_history = []
        self.no_improve_counts = [0, 0, 0]

    def update_scales(self, losses):
        """自适应调整loss缩放因子"""
        current_losses = torch.tensor([loss.item() for loss in losses])

        if len(self.loss_history) > 5:
            recent_avg = torch.tensor([sum(h[i] for h in self.loss_history[-5:]) / 5 for i in range(len(losses))])

            for i, (curr, avg) in enumerate(zip(current_losses, recent_avg)):
                if curr > avg * 1.05:
                    self.no_improve_counts[i] += 1
                    if self.no_improve_counts[i] >= self.patience:
                        self.scales[i] *= self.scale_factor
                        self.no_improve_counts[i] = 0
                else:
                    self.no_improve_counts[i] = 0

        self.loss_history.append(current_losses.tolist())
        return self.scales


class CombinedTTLTentTrainer(Seq2SeqTrainer):
    """Combined TTL-TENT Trainer for Test-Time Learning.

    This trainer combines:
    - TTL (Test-Time Learning): Optimizes perplexity/NLL on input sequences
    - TENT (Test-Time Entropy Minimization): Minimizes entropy on generated sequences

    The combined approach leverages both methods' strengths:
    - TTL helps the model adapt to the input distribution
    - TENT reduces uncertainty in generation
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.finetuning_args = finetuning_args

        # Combined token log: stores both perplexity and entropy information
        self.token_log = []

        # Initialize a counter for cumulative selected samples for logging.
        self.cumulative_selected_samples = 0

        # Initialize loss balancer
        self._initialize_loss_balancer()

        # Store original model for KL regularization
        if getattr(self.finetuning_args, "use_kl_regularization", False):
            self.original_model_state = None

        self.processing_class.padding_side = "left"

    def _initialize_loss_balancer(self):
        """初始化loss平衡器"""
        method = getattr(self.finetuning_args, "loss_balancing_method", "static")

        if method == "dynamic_weight":
            self.loss_balancer = DynamicWeightBalancer()
        elif method == "gradient_magnitude":
            self.loss_balancer = GradientMagnitudeBalancer()
        elif method == "uncertainty":
            self.loss_balancer = UncertaintyWeightBalancer()
            # 将balancer注册为模块的一部分
            if hasattr(self, "model") and self.model is not None:
                self.model.loss_balancer = self.loss_balancer
        elif method == "moving_average":
            self.loss_balancer = MovingAverageBalancer()
        elif method == "adaptive_scaling":
            self.loss_balancer = AdaptiveLossScaler()
        else:
            self.loss_balancer = None

    def _should_use_alternating_mode(self):
        """判断是否使用交替训练模式"""
        return getattr(self.finetuning_args, "alternating_training", False)

    def _get_current_training_mode(self):
        """获取当前训练模式 (奇数步: TTL, 偶数步: TENT)"""
        if not self._should_use_alternating_mode():
            return "combined"

        # 奇数步优化perplexity (TTL), 偶数步优化entropy (TENT)
        return "ttl" if (self.state.global_step % 2 == 1) else "tent"

    @override
    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
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

    # 实用性能优化 - 只需要修改这几个关键方法

    def _compute_kl_loss(self, model):
        """计算KL散度正则化损失 - 优化版"""
        # 🚀 优化1: 提前返回，避免不必要计算
        if not getattr(self.finetuning_args, "use_kl_regularization", False):
            return torch.tensor(0.0, device=model.device, requires_grad=False)

        # 🚀 优化2: 延迟初始化
        if self.original_model_state is None:
            self.original_model_state = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.original_model_state[name] = param.data.clone().detach()

        kl_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.original_model_state:
                diff = param - self.original_model_state[name]
                kl_loss = kl_loss + torch.sum(diff * diff)  # 更高效的平方计算

        return kl_loss

    @override
    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        # Get configuration parameters
        generation_len = getattr(self.finetuning_args, "generation_len", 0)
        ttl_weight = getattr(self.finetuning_args, "loss_weight_ttl", 1.0)
        tent_weight = getattr(self.finetuning_args, "loss_weight_tent", 1.0)
        kl_weight = getattr(self.finetuning_args, "kl_weight", 0.1)
        use_kl = getattr(self.finetuning_args, "use_kl_regularization", False)

        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
        else:
            raise RuntimeError("CombinedTrainer.compute_loss: No input_ids found in inputs.")

        # 获取当前训练模式
        training_mode = self._get_current_training_mode()

        # 🚀 优化3: 根据模式决定需要计算的loss
        need_ttl = training_mode in ["combined", "ttl"]
        need_tent = training_mode in ["combined", "tent"] and generation_len > 0

        # Initialize losses
        ttl_loss = torch.tensor(0.0, device=model.device, requires_grad=need_ttl)
        tent_loss = torch.tensor(0.0, device=model.device, requires_grad=need_tent)
        kl_loss = torch.tensor(0.0, device=model.device, requires_grad=use_kl)

        # 用于日志的变量
        per_sample_nll = torch.tensor([0.0], device=model.device)
        per_sample_ppl = torch.tensor([1.0], device=model.device)
        per_token_nll_input = torch.tensor([0.0], device=model.device)
        mask = torch.tensor([True], device=model.device).bool()
        shift_labels = torch.tensor([0], device=model.device)
        num_selected = 0
        total_samples = 1
        outputs = None

        # Part 1: TTL Loss on Input Sequence
        if need_ttl:
            # Forward pass for input sequence
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**model_inputs)
            logits = outputs["logits"]  # [B, L, V]

            # Compute TTL loss (perplexity/NLL based)
            shift_logits = logits[..., :-1, :].contiguous()  # [B, L-1, V]
            shift_labels = input_ids[..., 1:].contiguous()  # [B, L-1]

            loss_fct = CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())  # [B, L-1]

            per_token_nll_input = per_token_loss

            mask = shift_labels != IGNORE_INDEX  # [B, L-1]
            per_token_loss = per_token_loss * mask

            token_counts = mask.sum(dim=1).clamp(min=1)  # [B]
            per_sample_nll = per_token_loss.sum(dim=1) / token_counts

            # Parse TTL loss configuration
            ttl_loss_cfg = getattr(self.finetuning_args, "ttl_loss", "ppl_nll")
            parts = ttl_loss_cfg.split("_")
            metric_form = parts[0].lower()
            selection_form = parts[1].lower() if len(parts) == 2 else None

            # 🚀 优化4: 只在需要时才计算昂贵的exp操作
            per_sample_ppl = None
            if metric_form == "ppl" or selection_form == "ppl":
                per_sample_ppl = torch.exp(per_sample_nll.clamp(max=20))

            total_samples = per_sample_nll.size(0)

            # Apply selection gating if configured
            selection_score = None
            if selection_form is not None:
                scaler = getattr(self.finetuning_args, "ttl_sample_efficiency_scaler", 1.0)
                base_threshold = getattr(self.finetuning_args, "ttl_threshold", 3.0)

                if selection_form == "ppl":
                    if per_sample_ppl is None:
                        per_sample_ppl = torch.exp(per_sample_nll.clamp(max=20))
                    ppl_threshold = math.exp(base_threshold)
                    indicator = (per_sample_ppl > ppl_threshold).float()
                    raw_score = (per_sample_ppl / ppl_threshold).clamp(max=1e6)
                else:  # "nll"
                    nll_threshold = base_threshold
                    indicator = (per_sample_nll > nll_threshold).float()
                    raw_score = (per_sample_nll / nll_threshold).clamp(max=1e6)

                selection_score = (scaler * raw_score * indicator).detach()  # [B]
                num_selected = int(indicator.sum().item())

                # 🚀 优化5: 减少日志频率
                if self.state.global_step % (self.args.logging_steps * 5) == 0:
                    logger.info_rank0(f"[TTL] Selected {num_selected} / {total_samples} high-perplexity samples.")

            # Compute TTL loss
            if metric_form == "ppl":
                if per_sample_ppl is None:
                    per_sample_ppl = torch.exp(per_sample_nll.clamp(max=20))
                per_sample_metric = per_sample_ppl
            else:
                per_sample_metric = per_sample_nll

            if selection_score is None:
                ttl_loss = per_sample_metric.mean()
            else:
                ttl_loss = (selection_score * per_sample_metric).mean()

        # Part 2: TENT Loss on Generated Sequence
        generated_tokens = None
        entropy = None
        entropy_mask = None

        if need_tent:
            # 🚀 优化6: 避免不必要的模式切换
            was_training = model.training
            if was_training:
                self.model.eval()

            with torch.no_grad():
                generated_tokens = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=generation_len,
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                    do_sample=False,
                    top_k=None,
                    top_p=None,
                    temperature=1.0,
                )
                prompt_len = input_ids.size(1)

            if was_training:
                self.model.train()

            # Forward pass on generated sequence with gradient
            gen_outputs: CausalLMOutput = self.model(input_ids=generated_tokens)
            gen_logits = gen_outputs.logits  # [B, T, V]

            # Slice logits/tokens for generation part
            if getattr(self.finetuning_args, "use_full_entropy_in_generation", False):
                entropy_logits = gen_logits[:, :-1, :]
                entropy_tokens = generated_tokens[:, 1:]
            else:
                entropy_logits = gen_logits[:, prompt_len - 1 : -1, :]
                entropy_tokens = generated_tokens[:, prompt_len:]

            # 🚀 优化7: 更高效的熵计算
            log_probs = torch.nn.functional.log_softmax(entropy_logits, dim=-1)
            probs = torch.exp(log_probs)
            entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, L]

            # Mask padding
            entropy_mask = (entropy_tokens != self.processing_class.pad_token_id).float()

            # Compute TENT loss
            if getattr(self.finetuning_args, "use_emft_loss", False):
                loss_per_sequence = (entropy * entropy_mask).sum(dim=1)
                tent_loss = loss_per_sequence.mean()
            else:
                tent_loss = (entropy * entropy_mask).sum() / (entropy_mask.sum() + 1e-8)

        # Part 3: KL Regularization Loss - 使用优化版本
        if use_kl:
            kl_loss = self._compute_kl_loss(model)

        # 🚀 优化8: 只在combined模式且需要时才使用loss balancer
        balancer_freq = getattr(self.finetuning_args, "balancer_update_freq", 1)
        if (
            self.loss_balancer is not None
            and training_mode == "combined"
            and self.state.global_step % balancer_freq == 0
        ):
            method = getattr(self.finetuning_args, "loss_balancing_method", "static")

            # 只包含需要梯度的loss
            active_losses = [l for l in [ttl_loss, tent_loss, kl_loss] if l.requires_grad]

            if len(active_losses) > 1 and method == "gradient_magnitude":
                try:
                    weights = self.loss_balancer.compute_balanced_weights(model, active_losses)
                    idx = 0
                    if ttl_loss.requires_grad:
                        ttl_weight = weights[idx].item()
                        idx += 1
                    if tent_loss.requires_grad:
                        tent_weight = weights[idx].item()
                        idx += 1
                    if kl_loss.requires_grad:
                        kl_weight = weights[idx].item()
                except Exception:
                    # 如果梯度计算失败，使用默认权重
                    pass

        # 在交替训练模式下设置权重
        if training_mode == "ttl":
            tent_weight = 0.0
        elif training_mode == "tent":
            ttl_weight = 0.0

        # 🚀 优化9: 智能的loss组合，避免零权重计算
        total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        if ttl_weight > 0 and ttl_loss.requires_grad:
            total_loss = total_loss + ttl_weight * ttl_loss
        if tent_weight > 0 and tent_loss.requires_grad:
            total_loss = total_loss + tent_weight * tent_loss
        if kl_weight > 0 and kl_loss.requires_grad:
            total_loss = total_loss + kl_weight * kl_loss

        # 🚀 优化10: 减少日志计算和字符串操作
        if self.state.global_step % self.args.logging_steps == 0:
            if self.is_in_train:
                self.cumulative_selected_samples += num_selected

            # 简化日志输出
            logger.info_rank0(
                f"[S{self.state.global_step}] TTL={ttl_loss.item():.3f} "
                f"TENT={tent_loss.item():.3f} KL={kl_loss.item():.3f} "
                f"Total={total_loss.item():.3f} ({training_mode})"
            )

            # 批量构建日志字典
            log_data = {
                "ttl_loss": ttl_loss.item(),
                "tent_loss": tent_loss.item(),
                "kl_loss": kl_loss.item(),
                "batch_avg_nll": per_sample_nll.mean().item(),
                "training_mode": training_mode,
                "cumulative_selected_samples": float(self.cumulative_selected_samples),
            }

            if per_sample_ppl is not None:
                log_data["batch_avg_ppl"] = per_sample_ppl.mean().item()

            self.log(log_data)

        # 🚀 优化11: 大幅减少token级日志频率
        token_log_freq = getattr(self.finetuning_args, "token_log_freq", 50)
        if self.is_in_train and self.state.global_step % (self.args.logging_steps * token_log_freq) == 0:
            log_entry = {
                "input": {
                    "tokens": shift_labels.detach().cpu(),
                    "nll": per_token_nll_input.detach().cpu(),
                    "mask": mask.detach().cpu().bool(),
                }
            }

            if generated_tokens is not None and entropy is not None:
                log_entry["generated"] = {
                    "tokens": entropy_tokens.detach().cpu(),
                    "entropy": entropy.detach().cpu(),
                    "mask": entropy_mask.detach().cpu().bool(),
                }

            self.token_log.append(log_entry)

        # Store individual losses
        if not hasattr(self, "_custom_losses"):
            self._custom_losses = {"ttl_loss": [], "tent_loss": [], "kl_loss": []}
        self._custom_losses["ttl_loss"].append(ttl_loss.item())
        self._custom_losses["tent_loss"].append(tent_loss.item())
        self._custom_losses["kl_loss"].append(kl_loss.item())

        return (total_loss, outputs) if return_outputs else total_loss
