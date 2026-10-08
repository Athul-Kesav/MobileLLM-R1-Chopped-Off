"""Configuration for a MobileLLM-shaped Llama stack with ShiShuLM blocks."""

from __future__ import annotations

from transformers.configuration_utils import PreTrainedConfig


class ShishuLlamaConfig(PreTrainedConfig):
    """Hybrid decoder stack: full Llama blocks, then MLP-only ShiShu blocks.

    ``num_decoder_front`` full blocks are followed by ``num_mlp`` MLP-only
    blocks and then ``num_decoder_end`` full blocks. When ``num_mlp_shared``
    is greater than zero, that many adjacent MLP-only pairs share weights.
    """

    model_type = "shishu-llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 128256
    hidden_size: int = 576
    intermediate_size: int = 2048
    intermediate_size_mlp: int | None = None
    num_hidden_layers: int = 16
    num_decoder_front: int = 11
    num_decoder_end: int = 0
    num_mlp: int = 5
    num_mlp_shared: int = 0
    num_attention_heads: int = 9
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 128000
    eos_token_id: int | list[int] | None = None
    pretraining_tp: int = 1
    tie_word_embeddings: bool = True
    share_embeddings: bool = True
    share_embedding: bool = True
    use_qk_norm: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    rope_parameters: dict | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.intermediate_size_mlp is None:
            self.intermediate_size_mlp = self.intermediate_size
        if 2 * self.num_mlp_shared > self.num_mlp:
            raise ValueError(
                "num_mlp_shared counts adjacent MLP pairs, so "
                f"2 * num_mlp_shared ({2 * self.num_mlp_shared}) cannot exceed "
                f"num_mlp ({self.num_mlp})."
            )
        super().__post_init__(**kwargs)
        if self._attn_implementation is None:
            self._attn_implementation = "sdpa"

    @property
    def stack_size(self) -> int:
        return self.num_decoder_front + self.num_mlp + self.num_decoder_end


__all__ = ["ShishuLlamaConfig"]
