import math
from typing import Any, Dict, Iterable, Union

import torch
from torch.optim.optimizer import Optimizer

from utils import (
    stable_randn,
    stable_uniform,
    stable_discrete,
    stable_discrete_3,
    next_seed,
    split_seed,
)

def _down_proj(seed, rank, tensor, rand_func=stable_randn):
    lseed, rseed = split_seed(seed)
    if tensor.shape[0] < tensor.shape[-1]:
        left_projection = rand_func(
            (rank, tensor.shape[0]),
            seed=lseed,
            device=tensor.device,
            dtype=tensor.dtype,
        ) / math.sqrt(rank)

        return left_projection @ tensor
    else:
        right_projection = rand_func(
            (tensor.shape[-1], rank),
            seed=rseed,
            device=tensor.device,
            dtype=tensor.dtype,
        ) / math.sqrt(rank)
        return tensor @ right_projection

def _up_proj(seed, rank, shape, ctensor, rand_func=stable_randn):
    lseed, rseed = split_seed(seed)
    if shape[0] < shape[-1]:
        left_projection = rand_func(
            (rank, shape[0]),
            seed=lseed,
            device=ctensor.device,
            dtype=ctensor.dtype,
        ) / math.sqrt(rank)
        return left_projection.t() @ ctensor
    else:
        right_projection = rand_func(
            (shape[-1], rank),
            seed=rseed,
            device=ctensor.device,
            dtype=ctensor.dtype,
        ) / math.sqrt(rank)
        return ctensor @ right_projection.t()

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
            rand_distribution: str = 'normal',
    ) -> None:

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "rank": rank,
            "kappa": kappa,

        }
        super().__init__(params, defaults)

        if rand_distribution == 'normal':
            self.rand_func = stable_randn
        elif rand_distribution == 'uniform':
            self.rand_func = stable_uniform
        elif rand_distribution == 'discrete':
            self.rand_func = stable_discrete
        elif rand_distribution == 'discrete_3':
            self.rand_func = stable_discrete_3
        else:
            raise NotImplementedError

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

        compressed_grads: list[dict[str, Any]] = []

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
                    _current_seed = state["seed"]

                    cgrad = _down_proj(seed=_current_seed, rank=group["rank"], tensor=grad, rand_func=self.rand_func)
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

                    p.addcdiv_(_up_proj(seed=_current_seed, rank=group["rank"], shape=grad_shape, ctensor=corrected_avg, rand_func=self.rand_func),
                               denom,
                               value=-lr)

                    compressed_grads.append({
                        'grad': grad,
                        'cgrad': cgrad,
                        'compressed': True,
                    })

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

                    compressed_grads.append({
                        'grad': grad,
                        'cgrad': grad,
                        'compressed': False,
                    })

        return loss, compressed_grads
