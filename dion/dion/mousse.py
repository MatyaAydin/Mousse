import math
import torch
import torch.nn as nn
import torch.distributed as dist
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union, Dict

from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    pad_state_batch,
    to_local,
)
from .scalar_opts import adamw_update_foreach, lion_update_foreach


class Mousse(Optimizer):
    """
    Distributed Mousse optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu: Momentum factor for Muon algorithm.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float) -> Tensor`.
        shampoo_epsilon: The damping factor applied during spectral decomposition, making sure all eigenvalues 
            are at least equal to epsilon.
        shampoo_beta: The EMA coefficient of preconditioner.
        shampoo_update_freq: Update eigen values and vectors every T steps (default to 10). 
        shampoo_alpha: Curvature exponent. The smaller the value, the weaker the applied whitening intensity.
        LR_correction: Whether to apply bias correction in EMA of preconditioner.
        apply_norm: Whether to use grafting.
        use_LorR: 0. Double-sided preconditioner; 1. Left-only preconditioner; 2. Right-only preconditioner
        optimizer_debug: Whether to collect various debug information during the optimization process. This would incur additional time costs.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    """

    def __init__(
        self,
        params: ParamsT,
        model: nn.Module,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        newton_schulz_func: Optional[Callable] = None,
        shampoo_epsilon: float = 1e-10,
        shampoo_beta: float = 0.95,
        shampoo_update_freq: int = 10,
        shampoo_alpha: float = 0.125,
        LR_correction: bool = True,
        apply_norm: bool = True,
        use_LorR: int = 0,
        optimizer_debug: bool = False,
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", "None", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        self.apply_norm = apply_norm
        self.use_LorR = use_LorR

        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            algorithm="mousse",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
            shampoo_epsilon=shampoo_epsilon,
            shampoo_beta=shampoo_beta,
            shampoo_update_freq=shampoo_update_freq,
            shampoo_alpha=shampoo_alpha,
            LR_correction=LR_correction,
        )
        super().__init__(params, defaults)

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh is supported, but got {distributed_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh

        # Newton-Schulz configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(
                    f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}"
                )
            self._newton_schulz_func = newton_schulz_func
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5

        # Debug
        self.optimizer_debug = optimizer_debug
        if self.optimizer_debug:
            self._param_to_name = {
                id(p): name for name, p in model.named_parameters()
            }

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        debug_logs = {} if self.optimizer_debug else None

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        mousse_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "mousse":
                mousse_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        mousse_tasks = self._create_mousse_tasks(mousse_groups, debug_logs=debug_logs)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(mousse_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        # debug
        if self.optimizer_debug:
            # Gather debug logs from all devices to ensure complete logging on each rank
            if self._world_size > 1 and dist.is_initialized():
                all_debug_logs = [None for _ in range(self._world_size)]
                dist.all_gather_object(all_debug_logs, debug_logs, group=self._process_group)
                
                # Merge logs from all ranks
                merged_logs = {}
                for logs in all_debug_logs:
                    if logs:
                        merged_logs.update(logs)
                
                # Update the local debug_logs with the merged version
                debug_logs.update(merged_logs)

            return loss, debug_logs
        else:
            return loss, None

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
            elif algo == "mousse":
                if param.ndim == 2:
                    m, n = param.shape
                    if self.use_LorR in (0, 1):
                        state["L"] = torch.zeros(m, m, device=param.device, dtype=torch.float32)
                    if self.use_LorR in (0, 2):
                        state["R"] = torch.zeros(n, n, device=param.device, dtype=torch.float32)

                    if self.use_LorR in (0, 1):
                        state['eig_L'] = (
                            torch.zeros(m, device=param.device, dtype=torch.float32),
                            torch.eye(m, device=param.device, dtype=torch.float32)
                        )
                    if self.use_LorR in (0, 2):
                        state['eig_R'] = (
                            torch.zeros(n, device=param.device, dtype=torch.float32),
                            torch.eye(n, device=param.device, dtype=torch.float32)
                        )
                else:
                    raise NotImplementedError(f"Shampoo does not support parameters with {param.ndim} dimensions.")
        return state

    def _create_mousse_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "mousse",
        debug_logs: Optional[Dict] = None,
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of Muon matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Muon optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Wrap hyperparameters in tensors for torch.compile
            mousse_update_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                shampoo_beta=torch.tensor(group["shampoo_beta"]),
                shampoo_epsilon=torch.tensor(group["shampoo_epsilon"]),
                shampoo_update_freq=group["shampoo_update_freq"],
                shampoo_alpha=group["shampoo_alpha"],
                LR_correction=group["LR_correction"],
                nesterov=group["nesterov"],
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
                step=group["step"],
                apply_norm=self.apply_norm,
                use_LorR=self.use_LorR,
            )

            # Create batches of parameters of size self._world_size
            for params in create_param_batches(
                group_params, batch_size=self._world_size
            ):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]
                momentums = [state["momentum"] for state in states]

                # debug
                param_names = None
                if self.optimizer_debug:
                    param_names = [self._param_to_name.get(id(p), f"unnamed_param_{i}") for i, p in enumerate(params)]
                    if len(param_names) < self._world_size:
                        padding_needed = self._world_size - len(param_names)
                        param_names.extend([f"padded_param.{j}" for j in range(padding_needed)])

                # Get sharding state for DTensor
                is_batch_sharded = False
                is_matrix_sharded = False
                sharded_mesh_dim = None
                sharded_tensor_dim = None

                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must create optimizer with DeviceMesh if using DTensor parameters."
                        )

                    # Find the sharded placement and get its mesh and tensor dimensions
                    # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]

                    # If we don't flatten 3D matrices, we can ignore shard placements along batch dimensions
                    # Only keep placements that shard one of the two matrix dimensions
                    if not group["flatten"]:
                        matrix_dims = {params[0].ndim - 1, params[0].ndim - 2}
                        is_batch_sharded = any(
                            p.dim not in matrix_dims for _, p in shard_placements
                        )
                        shard_placements = [
                            (i, p) for i, p in shard_placements if p.dim in matrix_dims
                        ]

                    # Check that we have no more than 1 sharded matrix dimension
                    # Note that non-flattened 3D tensors can have additional sharded batch dimensions
                    # Flattened 3D tensors are limited to one sharded dimension out of all dimensions
                    if len(shard_placements) == 1:
                        is_matrix_sharded = True
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "Muon does not support parameters with multiple sharded dimensions."
                        )

                    # Check that the sharded mesh dimension matches optimizer's device mesh
                    if (
                        sharded_mesh_dim is not None
                        and params[0].device_mesh.get_group(sharded_mesh_dim)
                        != self._process_group
                    ):
                        raise RuntimeError(
                            f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh. "
                            f"DTensor has mesh: {params[0].device_mesh}, placements: {params[0].placements}, but optimizer was created with mesh: {self._distributed_mesh}."
                        )

                # Special case for 3D tensors sharded along batch dimension
                # As long as matrix dimensions are not sharded, each device will have whole matrices
                # Each device already has different matrices of the batch, so we can't parallelize further
                if is_batch_sharded and not is_matrix_sharded:
                    for x, g, m, s in zip(params, gradients, momentums, states):
                        yield AsyncTask(
                            mousse_update_batch_async(
                                X=[x],
                                G=[g],
                                M=[m],
                                states=[s],
                                shard_dim=None,  # No sharded matrix dim
                                param_names=param_names,
                                debug_logs=debug_logs,
                                optimizer_debug=self.optimizer_debug
                                **mousse_update_args,
                            )
                        )
                # Otherwise, we parallelize the Muon update across devices
                else:
                    yield AsyncTask(
                        mousse_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            states=pad_state_batch(states, self._world_size),
                            shard_dim=sharded_tensor_dim,
                            param_names=param_names,
                            debug_logs=debug_logs,
                            optimizer_debug=self.optimizer_debug,
                            **mousse_update_args,
                        )
                    )

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                )
            )


def clean_eigenvalues(evals, epsilon):
    """
    $\lambda := \lambda - \lambda_{min} + \epsilon$
    Ensure all eigenvalues are non-negative, with a minimum value of epsilon
    """
    min_eig = evals.min()

    shift = torch.maximum(torch.tensor(0.0, device=evals.device), -min_eig)
    shift += epsilon

    return evals + shift


def mousse_update_batch_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentums
    states: List[Dict], 
    lr: Tensor,  # Learning rate (scalar tensor)
    momentum: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    shampoo_beta: Tensor,
    shampoo_epsilon: Tensor,
    shampoo_update_freq: int,
    shampoo_alpha: float,
    LR_correction: bool,
    step: int,
    nesterov: bool,  # Whether to use Nesterov momentum
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    shard_dim: Optional[int] = None,  # Shard dimension for DTensor (if applicable)
    param_names: Optional[List[str]]= None,
    debug_logs: Optional[Dict] = None,
    optimizer_debug: bool = False,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    apply_norm: bool = False,
    use_LorR: int = 0,
) -> Generator[None, None, None]:
    """
    Batched version of Muon update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.
    """
    # Some state may be padded
    effective_length = len([s['L'] for s in states if s['L'] is not None])

    if optimizer_debug:
        param_name = param_names[device_rank]

    assert len(X) == len(G)
    assert len(X) == len(M)

    G = to_local(G)
    X = to_local(X)
    # Update momentum and compute the inputs for orthogonalization
    U = muon_update_pre_orthogonalize(
        G=G,
        M=to_local(M),
        momentum=momentum,
        nesterov=nesterov,
    )

    def apply_soap_and_ns(M, G, state, step, shampoo_beta = 0.95):
        G = G.to(torch.float32, copy=False)
        M_dtype = M.dtype
        M = M.to(torch.float32, copy=False)

        update_L, update_R = use_LorR in (0, 1), use_LorR in (0, 2)

        if update_L:
            m_dim = state['L'].shape[0]
        if update_R:
            n_dim = state['R'].shape[0]

        # 1. Update Curvature Statistics
        if update_L:
            state['L'].mul_(shampoo_beta).add_(G @ G.T, alpha=1-shampoo_beta)
        if update_R:
            state['R'].mul_(shampoo_beta).add_(G.T @ G, alpha=1-shampoo_beta)

        if optimizer_debug:
            if update_L:
                debug_logs[f"L_update/{param_name}"] = (G @ G.T).norm().item()
            if update_R:
                debug_logs[f"R_update/{param_name}"] = (G.T @ G).norm().item()

        # Calculate update direction adjustment every T steps
        if step % shampoo_update_freq == 1 or shampoo_update_freq == 1:
            if LR_correction:
                bias_correction = 1.0 - shampoo_beta ** step
                if update_L:
                    L_corrected = state["L"] / bias_correction
                if update_R:
                    R_corrected = state["R"] / bias_correction
            else:
                if update_L:
                    L_corrected = state["L"]
                if update_R:
                    R_corrected = state["R"]

            if optimizer_debug:
                if update_L:
                    rms_L = L_corrected.pow(2).mean().sqrt().item()
                    debug_logs[f"rms_L/{param_name}"] = rms_L
                if update_R:
                    rms_R = R_corrected.pow(2).mean().sqrt().item()
                    debug_logs[f"rms_R/{param_name}"] = rms_R

            # 2. trace normalization
            if update_L:
                trace_L = L_corrected.trace()
                L_corrected.mul_(m_dim / trace_L)
            if update_R:
                trace_R = R_corrected.trace()
                R_corrected.mul_(n_dim / trace_R)

            # 3. Eigen-decomposition
            if update_L:
                eval_L, evec_L = torch.linalg.eigh(
                    L_corrected + shampoo_epsilon * torch.eye(m_dim, device=G.device)
                )
            if update_R:
                eval_R, evec_R = torch.linalg.eigh(
                    R_corrected + shampoo_epsilon * torch.eye(n_dim, device=G.device)
                )

            if optimizer_debug:
                if update_L:
                    rms_L = L_corrected.pow(2).mean().sqrt().item()
                    debug_logs[f"rms_L_trace/{param_name}"] = rms_L
                if update_R:
                    rms_R = R_corrected.pow(2).mean().sqrt().item()
                    debug_logs[f"rms_R_trace/{param_name}"] = rms_R

            if optimizer_debug:
                if update_L:
                    debug_logs[f"eval_L_max/{param_name}"] = eval_L.max()
                    debug_logs[f"eval_L_min/{param_name}"] = eval_L.min()
                    debug_logs[f"cond_L/{param_name}"] = eval_L.max() / (eval_L.min() + 1e-30)
                if update_R:
                    debug_logs[f"eval_R_max/{param_name}"] = eval_R.max()
                    debug_logs[f"eval_R_min/{param_name}"] = eval_R.min()
                    debug_logs[f"cond_R/{param_name}"] = eval_R.max() / (eval_R.min() + 1e-30)

            if update_L:
                eval_L = clean_eigenvalues(eval_L, shampoo_epsilon)
            if update_R:
                eval_R = clean_eigenvalues(eval_R, shampoo_epsilon)

            if optimizer_debug:
                if update_L:
                    debug_logs[f"eval_L_max_correction/{param_name}"] = eval_L.max()
                    debug_logs[f"eval_L_min_correction/{param_name}"] = eval_L.min()
                    debug_logs[f"cond_L_correction/{param_name}"] = eval_L.max() / (eval_L.min() + 1e-30)
                if update_R:
                    debug_logs[f"eval_R_max_correction/{param_name}"] = eval_R.max()
                    debug_logs[f"eval_R_min_correction/{param_name}"] = eval_R.min()
                    debug_logs[f"cond_R_correction/{param_name}"] = eval_R.max() / (eval_R.min() + 1e-30)
            
            # Save eigen values and vectors
            if update_L:
                state['eig_L'] = (eval_L, evec_L)
            if update_R:
                state['eig_R'] = (eval_R, evec_R)
 
        # 4. Whitening ($\tilde{M} = L^{-\alpha} M R^{-\alpha}$)
        if update_L:
            eval_L, evec_L = state["eig_L"]
        if update_R:
            eval_R, evec_R = state["eig_R"]

        # 4.1. rotate
        if update_L:
            M = evec_L.T @ M
        if update_R:
            M = M @ evec_R

        if optimizer_debug:
            debug_logs[f"rms_U_before_rotation/{param_name}"] = M.pow(2).mean().sqrt().item()

        # 4.2. scale
        if update_L:
            eval_L_sqrt = (eval_L.abs()).pow(shampoo_alpha)
        if update_R:
            eval_R_sqrt = (eval_R.abs()).pow(shampoo_alpha)

        if update_L:
            M = M / eval_L_sqrt.unsqueeze(1)
        if update_R:
            M = M / eval_R_sqrt.unsqueeze(0)

        if optimizer_debug:
            debug_logs[f"rms_U_before_scale/{param_name}"] = M.pow(2).mean().sqrt().item()

        # 5. Spectral Constraint
        M = muon_update_newton_schulz(
            M,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )
        # 5.1. Save the norm
        if apply_norm:
            M_norm = M.norm()

        if optimizer_debug:
            debug_logs[f"rms_U_ns/{param_name}"] = M.pow(2).mean().sqrt().item()

        # 6. Unwhitening ($M = L^{-\alpha} \tilde{M} R^{-\alpha}$)
        # 6.1. scale
        if update_L:
            M = M / eval_L_sqrt.unsqueeze(1)
        if update_R:
            M = M / eval_R_sqrt.unsqueeze(0)

        if optimizer_debug:
            debug_logs[f"rms_U_after_scale/{param_name}"] = M.pow(2).mean().sqrt().item()

        # 6.2. rotate
        if update_L:
            M = evec_L @ M
        if update_R:
            M = M @ evec_R.T

        if optimizer_debug:
            debug_logs[f"rms_U_after_rotation/{param_name}"] = M.pow(2).mean().sqrt().item()

        # 6.3. grafting
        if apply_norm:
            if optimizer_debug:
                debug_logs[f"norm_U_ns/{param_name}"] = M_norm.item()
                debug_logs[f"norm_U_after_rotation/{param_name}"] = M.norm().item()
                debug_logs[f"norm_scale_U_after_rotation/{param_name}"] = (M_norm / M.norm()).item()
            M = (M_norm / M.norm()) * M

        return M.to(M_dtype)

    # Get one whole matrix for each device to orthogonalize
    if shard_dim is not None:
        # Use all-to-all to transform from a batch of shards to a single whole matrix
        # https://www.essential.ai/blog/infra
        assert 0, "Not fully tested, as all experiments in the paper were conducted on one single cluster with DDP."
        assert len(X) == world_size, "Batch size must equal world size"
        assert (
            process_group is not None
        ), "process_group must be provided for sharded DTensors"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert not isinstance(U[0], DTensor), "U should contain local shards"
        assert (
            X[0].size(shard_dim) % world_size == 0
        ), f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}."

        # Allocate buffers to receive shards of one whole matrix from other devices
        single_matrix_shards = [torch.empty_like(u) for u in U]

        # Redistribute the shards to form one unique full tensor on each device
        work = dist.all_to_all(
            single_matrix_shards, U, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Concatentate shards to form a whole matrix to orthogonalize
        single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Split result back into shards
        # Contiguous is needed for all-to-all to work correctly
        single_matrix_shards = [
            x.contiguous()
            for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
        ]

        # Redistribute the orthogonalized tensor back to original layout
        work = dist.all_to_all(
            U, single_matrix_shards, group=process_group, async_op=True
        )
        yield
        work.wait()

    # Matrices are not sharded, so we can distribute the batch across different devices
    # Get a single matrix of the batch corresponding to this device
    elif len(U) > 1:
        # (zyc): Only Consider this situation now
        assert len(U) == world_size, "Batch size must equal world size"
        assert process_group is not None

        # Each rank only process their corresponding parts
        U_rank = U[device_rank]
        assert not isinstance(U_rank, DTensor)

        G_rank = G[device_rank]
        state = states[device_rank]

        if device_rank >= effective_length:
            U_rank = U_rank
        else:
            U_rank = apply_soap_and_ns(
                U_rank,
                G_rank,
                state,
                step=step,
                shampoo_beta=shampoo_beta,
            )

        # Allocate empty tensors to receive updates from other devices
        U = [torch.empty_like(u) for u in U]

        # All gather orthogonalized results from other devices into buffer
        work = dist.all_gather(
            U, U_rank.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

    # Single tensor with no sharded dimension. This happens in 2 cases:
    # - Running on a single GPU
    # - 3D+ tensors sharded along a batch dimension (different whole matrices per device)
    else:
        assert len(U) == 1
        U[0] = apply_soap_and_ns(
            U[0],
            G[0],
            state,
            step=step,
            shampoo_beta=shampoo_beta,
        )

    # Compute scaled learning rate
    # Do this before to_local(X) because we use the full tensor shape, not the shard shape
    if adjust_lr is None or adjust_lr == 'None':
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    if debug_logs is not None and param_names is not None:
        lr_val = adjusted_lr.item() if isinstance(adjusted_lr, Tensor) else adjusted_lr
        for u, param_name in zip(U, param_names):
            if "padded_param" in param_name:
                continue
            # Only compute RMS for the param handled by this rank in distributed mode
            # Or for all params if not distributed (len(U) == 1)
            is_my_param = (len(U) == 1) or (param_names.index(param_name) == device_rank)

            if is_my_param:
                rms = u.to(torch.float32).pow(2).mean().sqrt().item()
                debug_logs[f"update_rms/{param_name}"] = rms * lr_val
                debug_logs[f"update_rms_wo_lr/{param_name}"] = rms
                debug_logs[f"update_lr/{param_name}"] = lr_val

    local_X = to_local(X)

    # Update model parameters with orthogonalized output
    muon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )


def adamw_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
) -> Generator[None, None, None]:
    """
    Async wrapper around foreach AdamW update.
    """
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)
    yield


def lion_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
) -> Generator[None, None, None]:
    """
    Async wrapper around foreach Lion update.
    """
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay)
    yield


@torch.compile(fullgraph=True)
def muon_update_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    momentum: Tensor,
    nesterov: bool,
) -> List[Tensor]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_mul_(M, momentum)
    torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U


@torch.compile(fullgraph=True)
def muon_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """
    Apply weight decay and weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Weight update
    U = torch._foreach_mul(U, adjusted_lr)
    torch._foreach_sub_(X, U)


def muon_update_newton_schulz(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Tensor:
    """
    Flatten the input tensor if needed and call the Newton-Schulz function.
    """
    original_shape = X.shape
    if flatten and X.ndim >= 3:
        # Flatten 3D+ tensors to 2D matrix
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        # Given 4D+ batch, flatten to 3D batch
        X = X.flatten(end_dim=-3)

    return newton_schulz_func(X, epsilon=epsilon).reshape(original_shape)


def adjust_lr_rms_norm(lr, param_shape, flatten):
    # Adjust learning rate for constant element-wise RMS norm
    # https://arxiv.org/abs/2502.16982
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    adjusted_ratio = 0.2 * math.sqrt(max(fan_out, fan_in))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


def adjust_lr_spectral_norm(lr, param_shape, flatten):
    # Adjust from spectral norm 1 to RMS operator norm 1
    # https://arxiv.org/abs/2310.17813
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    adjusted_lr = lr * math.sqrt(fan_out / fan_in)
    return adjusted_lr


@torch.compile(fullgraph=True)
def zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7):
    """
    Newton-Schulz iteration to approximate the orthogonalization of X.
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
