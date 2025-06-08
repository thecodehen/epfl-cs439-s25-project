import math
from typing import Any, Dict, Iterable, Union

import torch
from torch.optim.optimizer import Optimizer

from utils import stable_randn, next_seed, split_seed

class FloraAdam(Optimizer):
    def __init__(
            self,
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
            lr:float = None,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            rank: int = None,
            kappa: int = 1000,
            seed: int = 0,
    ) -> None:

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "rank": rank,
            "kappa": kappa,

        }
        super().__init__(params, defaults)

        params_idx = seed
        for group in self.param_groups:
            for p in group["params"]:
                params_idx += 1
                if p.requires_grad:
                    self.state[p]["seed"] = params_idx

    def _should_compress(self, param_group: Dict, param_shape: tuple[int, ...]) -> bool:
        factored = len(param_shape) == 2
        should_compress= (
                param_group["rank"] is not None
                and param_group["rank"] > 0
                and factored
                and min(param_shape) >= param_group["rank"]
                and max(param_shape) / min(param_shape) <= 4  # rule out embeddings
        )
        return should_compress

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, (beta1, beta2), eps = group['lr'], group['betas'], group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                grad_shape = grad.shape
                should_compress = self._should_compress(group, grad_shape)

                state = self.state[p]

                # State initialization
                if (state and'step' not in state) or (not state):
                    state['step'] = 0

                    if should_compress:
                        if grad_shape[0] < grad_shape[-1]:
                            cshape = (group["rank"], grad_shape[-1])
                        else:
                            cshape = (grad_shape[0], group["rank"])

                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros(cshape).to(grad)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    else:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                if should_compress:
                    def _down_proj(seed, rank, tensor):
                        lseed, rseed = split_seed(seed)
                        if tensor.shape[0] < tensor.shape[-1]:
                            left_projection = stable_randn(
                                (rank, tensor.shape[0]),
                                seed=lseed,
                                device=tensor.device,
                                dtype=tensor.dtype,
                            ) / math.sqrt(rank)

                            return left_projection @ tensor
                        else:
                            right_projection = stable_randn(
                                (tensor.shape[-1], rank),
                                seed=rseed,
                                device=tensor.device,
                                dtype=tensor.dtype,
                            ) / math.sqrt(rank)
                        return tensor @ right_projection

                    def _up_proj(seed, rank, shape, ctensor):
                        lseed, rseed = split_seed(seed)
                        if shape[0] < shape[-1]:
                            left_projection = stable_randn(
                                (rank, shape[0]),
                                seed=lseed,
                                device=ctensor.device,
                                dtype=ctensor.dtype,
                            ) / math.sqrt(rank)
                            return left_projection.t() @ ctensor
                        else:
                            right_projection = stable_randn(
                                (shape[-1], rank),
                                seed=rseed,
                                device=ctensor.device,
                                dtype=ctensor.dtype,
                            ) / math.sqrt(rank)
                            return ctensor @ right_projection.t()

                    _current_seed = state["seed"]

                    cgrad = _down_proj(seed=_current_seed, rank=group["rank"], tensor=grad)
                    # Update biased first moment estimate
                    exp_avg.mul_(beta1).add_(cgrad, alpha=1 - beta1)
                    # Update biased second raw moment estimate
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Compute bias-corrected moments
                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t
                    corrected_avg = exp_avg / bias_correction1
                    corrected_avg_sq = exp_avg_sq / bias_correction2

                    # Parameter update
                    denom = corrected_avg_sq.sqrt().add_(eps)

                    p.addcdiv_(_up_proj(seed=_current_seed, rank=group["rank"], shape=grad_shape, ctensor=corrected_avg),
                               denom,
                               value=-lr)

                    # Time for a new seed
                    if state["step"] % group["kappa"] == 0:
                        _next_seed = next_seed(state["seed"])

                        state["exp_avg"].copy_(
                            _down_proj(
                                seed=_next_seed,
                                rank=group["rank"],
                                tensor=_up_proj(
                                    seed=_current_seed,
                                    rank=group["rank"],
                                    shape=grad_shape,
                                    ctensor=state["exp_avg"]
                                ),
                            )
                        )

                        state["seed"] = _next_seed
                        _current_seed = _next_seed

                else:
                    # Update biased first moment estimate
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    # Update biased second raw moment estimate
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Compute bias-corrected moments
                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t
                    corrected_avg = exp_avg / bias_correction1
                    corrected_avg_sq = exp_avg_sq / bias_correction2

                    # Parameter update
                    denom = corrected_avg_sq.sqrt().add_(eps)
                    p.addcdiv_(corrected_avg, denom, value=-lr)

        return loss
