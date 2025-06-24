"""TEMPO algorithm components."""

# RoPE modifications
from .rope.position_mapper import PositionMapper
from .rope.embedding_modifier import (
    apply_rotary_pos_emb,
    modify_positions_for_parallel_tokens,
    RoPECache
)
from .rope.model_patcher import ModelPatcher

# Attention analysis
from .attention.mask_builder import AttentionMaskBuilder
from .attention.pattern_analyzer import AttentionPatternAnalyzer
from .attention.weight_extractor import AttentionWeightExtractor, AttentionWeights

# Generation components
from .generation.logits_processor import LogitsProcessor
from .generation.kv_cache_manager import KVCacheManager, KVCache
from .generation.parallel_processor import ParallelProcessor, ParallelTokenSet

# Pruning algorithms
from .pruning.attention_pruner import AttentionBasedPruner
from .pruning.threshold_manager import DynamicThresholdManager
from .pruning.multi_scale_pruner import MultiScaleAttentionPruner

__all__ = [
    # RoPE
    'PositionMapper',
    'apply_rotary_pos_emb',
    'modify_positions_for_parallel_tokens',
    'RoPECache',
    'ModelPatcher',
    
    # Attention
    'AttentionMaskBuilder',
    'AttentionPatternAnalyzer',
    'AttentionWeightExtractor',
    'AttentionWeights',
    
    # Generation
    'LogitsProcessor',
    'KVCacheManager',
    'KVCache',
    'ParallelProcessor',
    'ParallelTokenSet',
    
    # Pruning
    'AttentionBasedPruner',
    'DynamicThresholdManager',
    'MultiScaleAttentionPruner',
]