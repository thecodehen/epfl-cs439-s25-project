import numpy as np
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, Iterable, Optional, Sequence, Union
import torch
import math

def split_seed(seed: int) -> tuple[int, int]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randint(0, torch.iinfo(torch.int64).max, (2,), generator=generator, device=generator.device).tolist()
    )

def next_seed(seed: int, adv: int = 0xF) -> int:
    """
    This is a naive helper function to generate a new seed from the given seed.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, torch.iinfo(torch.int64).max, (adv,), generator=generator, device=generator.device
    ).tolist()[-1]

def stable_randn(
    shape: Union[int, Sequence[int]],
    seed: int,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    rn = torch.randn(shape, generator=generator, device=generator.device, dtype=dtype)
    return rn

class FloraAdamDecreasingRanks(Optimizer):
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr:float = None,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        rankFrom: int = None,
        rankTo: int = None,
        kappa: int = 1000,
        seed: int = 0,
        epochs: int = 100,
    ) -> None:

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "rank": rankFrom,
            "rankFrom": rankFrom,
            "rankTo": rankTo,
            "kappa": kappa,
            "epochs": epochs,

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
                        # Assign the Rank to the state also
                        state['rank'] = group["rank"]

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

                    cgrad = _down_proj(seed=_current_seed, rank=state["rank"], tensor=grad)
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

                    p.addcdiv_(_up_proj(seed=_current_seed, rank=state["rank"], shape=grad_shape, ctensor=corrected_avg),
                              denom,
                              value=-lr)

                    # Time for a new seed
                    if state["step"] % group["kappa"] == 0:
                        _new_rank = np.round(group["rankFrom"] - ((group["rankFrom"] - group["rankTo"]) * state["step"]/group['epochs'])).astype(int)

                        _next_seed = next_seed(state["seed"])

                        state["exp_avg"] = _down_proj(
                                seed=_next_seed,
                                rank=_new_rank,
                                tensor=_up_proj(
                                    seed=_current_seed,
                                    rank=state["rank"],
                                    shape=grad_shape,
                                    ctensor=state["exp_avg"]
                                    ),
                            )


                        state["seed"] = _next_seed
                        _current_seed = _next_seed
                        state["rank"] = _new_rank

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