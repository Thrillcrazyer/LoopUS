import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def _unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP / DataParallel / FSDP wrappers without touching torch.compile.

    ``accelerator.unwrap_model`` has a bug when the model is
    ``DDP(compiled_module)`` – ``has_compiled_regions`` fires on the
    DDP wrapper and then tries ``__dict__["_orig_mod"]`` which doesn't
    exist.  This helper simply peels DDP/DP/FSDP ``.module`` attributes.
    """
    while isinstance(
        model,
        (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel, FSDP),
    ):
        model = model.module
    return model