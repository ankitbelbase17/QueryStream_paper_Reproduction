"""QueryStream -- reproduction of the ICLR 2026 paper.

QueryStream: Advancing Streaming Video Understanding with Query-Aware Pruning
and Proactive Response (Zhang, Yang, Wang, Qian, Xu).

Milestone 1 surface: config, ops, encoder, DSH memory, QDP, RTAR, token buffer.
Milestone 2 adds model_wrapper.py (Qwen2.5-VL-7B-Instruct + JIT delivery).
Milestone 3 adds eval_streaming.py (StreamingBench, then OVO-Bench).
"""

__version__ = "0.1.0"

from .config import (
    BackboneConfig,
    BufferConfig,
    EncoderConfig,
    QDPConfig,
    QueryStreamConfig,
    RTARConfig,
)
from .ops import (
    build_mrope_coords,
    cosine_sim_rowwise,
    cosine_sim_to_vector,
    expand_mask_to_temporal_merge,
    keep_rate,
    l2_normalize,
    mean_pool_frame,
    resample_patch_mask,
)
from .encoder import EncodedFrame, OpenCLIPEncoder, SyntheticEncoder, VisionLanguageEncoder
from .dsh_memory import DSHMemory
from .qdp import QDPModule, QDPOutput
from .rtar import EventDebouncer, RelevanceCalibrator, RTARModule, RTAROutput
from .buffer import BufferedFrame, FlushResult, TokenBuffer

__all__ = [
    "__version__",
    "QueryStreamConfig", "EncoderConfig", "QDPConfig", "RTARConfig",
    "BufferConfig", "BackboneConfig",
    "l2_normalize", "cosine_sim_to_vector", "cosine_sim_rowwise", "mean_pool_frame",
    "build_mrope_coords", "resample_patch_mask", "expand_mask_to_temporal_merge",
    "keep_rate",
    "EncodedFrame", "VisionLanguageEncoder", "OpenCLIPEncoder", "SyntheticEncoder",
    "DSHMemory",
    "QDPModule", "QDPOutput",
    "RTARModule", "RTAROutput", "RelevanceCalibrator", "EventDebouncer",
    "TokenBuffer", "BufferedFrame", "FlushResult",
]
