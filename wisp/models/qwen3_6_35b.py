"""
wisp.models.qwen3_6_35b — Qwen3.6-35B-A3B
(35B total, ~3.6B active per token, GQA 32Q/8KV, 48 layers, 
 64 experts/layer, top-4 routing).

Expert lookups per token: 4 x 48 = 192.

ARCHITECTURE SUMMARY (based on Alibaba Qwen Team release):
  • Total parameters: ~35B
  • Active parameters per token: ~3.6B  
  • Dense path: attention + shared projections (~8GB)
  • MoE: 64 experts/layer, top-4 selection
  • Layers: 48 transformer blocks
  • Hidden size: 5120
  • Intermediate size: 17920 per expert
  • Attention: GQA with 32 query heads, 8 KV heads
  • Context: 256K tokens
  • Experts are int4 quantized in WBIN2 format

TENSOR NAMING — Qwen follows Llama-style naming:
  Dense  : model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
           model.layers.{i}.input_layernorm.weight / post_attention_...
           model.embed_tokens.weight / model.norm.weight / lm_head.weight
  Experts: model.layers.{i}.mlp.experts.{j}.{gate,up,down}_proj.weight
  Router : model.layers.{i}.mlp.gate.weight

The expert .bin layout is IDENTICAL to GLM/DeepSeek/Mixtral — 
120-byte header + canonical [gate | up | down] int4 blobs.
"""

from __future__ import annotations

import re

from .base_adapter import ModelAdapter
from . import constants as C


_QWEN_PROJ_MAP = {"gate_proj": "gate_proj", "up_proj": "up_proj", "down_proj": "down_proj"}


class Qwen3635BAdapter(ModelAdapter):

    # GQA geometry (used by the info display + manifest extras)
    num_attention_heads = 32
    num_kv_heads = 8              # 8 KV heads serve 32 query heads (x4)
    intermediate_size = 17920     # Per-expert FFN size
    max_position_embeddings = 262144  # 256K context

    @property
    def name(self) -> str: 
        return "Qwen3.6-35B-A3B"

    @property
    def family(self) -> str: 
        return "qwen3_6_35b"

    @property
    def hf_model_id(self) -> str: 
        return C.HF_MODEL_ID["qwen3_6_35b"]

    @property
    def total_parameters(self) -> int:
        return C.TOTAL_PARAMETERS["qwen3_6_35b"]

    @property
    def num_layers(self) -> int: 
        return C.NUM_LAYERS["qwen3_6_35b"]

    @property
    def num_experts_per_layer(self) -> int:
        return C.NUM_EXPERTS_PER_LAYER["qwen3_6_35b"]

    @property
    def num_shared_experts(self) -> int:
        return C.NUM_SHARED_EXPERTS["qwen3_6_35b"]   # Typically 0 for pure MoE

    @property
    def top_k_routing(self) -> int: 
        return C.TOP_K_ROUTING["qwen3_6_35b"]

    @property
    def expert_size_bytes(self) -> int:
        return C.EXPERT_SIZE_INT4["qwen3_6_35b"]

    @property
    def dense_layer_size_bytes(self) -> int:
        return C.DENSE_SIZE["qwen3_6_35b"]

    @property
    def hidden_size(self) -> int: 
        return C.HIDDEN_SIZE["qwen3_6_35b"]

    @property
    def attention_type(self) -> str:
        return C.ATTENTION_TYPE["qwen3_6_35b"]       # "GQA"

    @property
    def vocab_size(self) -> int: 
        return C.VOCAB_SIZE["qwen3_6_35b"]

    @property
    def has_native_mtp(self) -> bool: 
        return False  # No native MTP head announced

    @property
    def drafter_hf_id(self) -> str: 
        return C.DRAFTER_HF_ID["qwen3_6_35b"]

    @property
    def default_acceptance_rate(self) -> float:
        return C.DEFAULT_ACCEPTANCE_RATE["qwen3_6_35b"]

    @property
    def mtp_k(self) -> int:
        return 3  # Default draft length for speculative decoding

    def get_drafter_config(self) -> dict:
        return {
            "type": "same_family",
            "hf_id": self.drafter_hf_id,
            "dtype": "int4",
            "k": self.mtp_k,
            "acceptance": self.default_acceptance_rate,
        }

    # ------------------------------------------------------------------ #
    # Qwen tensor naming
    # ------------------------------------------------------------------ #
    @property
    def expert_weight_pattern(self) -> re.Pattern:
        return re.compile(
            r"model\.layers\.(?P<layer>\d+)\.mlp\.experts"
            r"\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight"
        )

    def normalize_expert_proj(self, proj: str) -> str:
        return _QWEN_PROJ_MAP[proj]

    def canonical_dense_name(self, hf_name: str) -> str | None:
        # Qwen uses standard Llama-style naming
        # Only the router key might need special handling
        m = re.match(
            r"model\.layers\.(\d+)\.mlp\.gate\.weight",
            hf_name)
        if m:
            return f"layers.{m.group(1)}.router"
        
        # Check for router bias
        m = re.match(
            r"model\.layers\.(\d+)\.mlp\.gate\.bias",
            hf_name)
        if m:
            return f"layers.{m.group(1)}.router_bias"
            
        return super().canonical_dense_name(hf_name)

    def is_dense_tensor(self, key: str) -> bool:
        return "mlp.experts" not in key

    def get_expert_key(self, layer_idx: int, expert_idx: int) -> list[str]:
        base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        return [f"{base}.gate_proj.weight",   # gate
                f"{base}.up_proj.weight",     # up
                f"{base}.down_proj.weight"]   # down

    def get_router_key(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}.mlp.gate.weight"
