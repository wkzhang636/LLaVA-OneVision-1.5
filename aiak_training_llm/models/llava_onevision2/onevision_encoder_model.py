import torch
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.spec_utils import ModuleSpec

from aiak_training_llm.models.llava_onevision2.llava_onevision2_config import (
    VisionConfig,
)
from aiak_training_llm.models.llava_onevision2.vision_transformer_block import TransformerBlock


class PatchEmbed(torch.nn.Module):
    """
    Image to Patch Embedding module.

    Converts input images into patch embeddings using a convolutional projection.

    Args:
        patch_size (int): Size of each square patch.  Default:  14.
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Dimension of patch embeddings. Default: 1024.
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Convolutional projection to extract patches
        self.proj = torch.nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to convert images to patch embeddings.

        Args:
            hidden_states (torch.Tensor): Input patches of shape [total_patches, channels * patch_size * patch_size].
        Returns:
            torch. Tensor: Patch embeddings of shape [num_patches, embed_dim].
        """
        # Reshape to [num_patches, in_channels, patch_size, patch_size]
        hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states).view(-1, self.embed_dim)
        return hidden_states


class VideoRotaryEmbeddingSplit466(torch.nn.Module):
    """
    3D Rotary Position Embedding for video inputs with 4: 6:6 dimension split.

    Constructs rotary frequencies for temporal (T), height (H), and width (W) dimensions
    with a 4:6:6 ratio split across head dimensions.

    Args:
        hidden_size (int): Total hidden dimension size.
        num_attention_heads (int): Number of attention heads.
        rope_theta (float): Base value for frequency calculation.
    """

    def __init__(self, hidden_size, num_attention_heads, rope_theta):
        super().__init__()
        head_dim = hidden_size // num_attention_heads
        base = rope_theta
        assert head_dim % 2 == 0, "head_dim must be even for rotary."
        assert head_dim % 16 == 0, "head_dim must be divisible by 16."
        half = head_dim // 2
        assert half % 16 == 0, "head_dim//2 must also be divisible by 16 to split into 4:6:6."

        self.head_dim = head_dim
        self.half = half

        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        self.register_buffer(
            "inv_freq_t",
            1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False,
        )

    def forward(self, t: int, h: int, w: int, device="cuda"):
        """
        Compute rotary position embeddings from grid_thw (Qwen2VL style).

        Args:
            grid_thw: [num_samples, 3] tensor with [t, h, w] for each sample

        Returns:
            freqs: [total_seq_len, half] tensor of position frequencies
        """

        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        # Compute frequency tables
        ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
        fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
        fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

        # Build position indices for this sample
        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)

        # Concatenate frequencies: [seq_len, half]
        sample_freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
        return sample_freqs

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings from explicit patch positions.

        Args:
            patch_positions: [seq_len, 3] tensor with [t, h, w] positions for each patch

        Returns:
            freqs: [seq_len, half] tensor of position frequencies
        """
        device = patch_positions.device
        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        t_pos = patch_positions[:, 0].float()
        h_pos = patch_positions[:, 1].float()
        w_pos = patch_positions[:, 2].float()

        ft = torch.outer(t_pos, inv_t)
        fh = torch.outer(h_pos, inv_h)
        fw = torch.outer(w_pos, inv_w)

        return torch.cat([ft, fh, fw], dim=-1)


def convert_rope_to_block_layout(
    freqs: torch.Tensor, t: int, h: int, w: int, spatial_merge_size: int = 2
) -> torch.Tensor:
    """
    Convert RoPE from row-major order (1x1 layout) to 2x2 block layout.

    The image processor arranges patches in 2x2 blocks when spatial_merge_size=2:
    - Row-major order: [p(0,0), p(0,1), p(0,2), p(0,3), ..., p(1,0), p(1,1), ...]
    - Block order: [p(0,0), p(0,1), p(1,0), p(1,1)], [p(0,2), p(0,3), p(1,2), p(1,3)], ...

    Args:
        freqs: RoPE frequencies in row-major order, shape [t*h*w, half]
        t: temporal dimension
        h: height (unmerged patch count)
        w: width (unmerged patch count)
        spatial_merge_size: size of spatial merge blocks (default: 2)

    Returns:
        torch.Tensor: RoPE frequencies in 2x2 block order, same shape [t*h*w, half]
    """
    sms = spatial_merge_size
    if sms == 1:
        return freqs

    half = freqs.shape[-1]

    # freqs shape: [t*h*w, half]
    # Reshape to [t, h, w, half]
    freqs = freqs.view(t, h, w, half)

    # Calculate merged dimensions
    h_merged = h // sms
    w_merged = w // sms

    # Reshape to [t, h_merged, sms, w_merged, sms, half]
    freqs = freqs.view(t, h_merged, sms, w_merged, sms, half)

    # Permute to [t, h_merged, w_merged, sms_h, sms_w, half] - 2x2 block order
    freqs = freqs.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Reshape back to [t*h*w, half]
    freqs = freqs.view(t * h * w, half)

    return freqs


def convert_positions_to_block_layout(
    positions: torch.Tensor, t: int, h: int, w: int, spatial_merge_size: int = 2
) -> torch.Tensor:
    """
    Convert patch positions from row-major order to 2x2 block layout.

    This function reorders patch positions to match the 2x2 block arrangement
    used by the image processor. Uses index-based reordering instead of reshape.

    Args:
        positions: Patch positions in row-major order, shape [t*h*w, 3]
        t: temporal dimension
        h: height (unmerged patch count)
        w: width (unmerged patch count)
        spatial_merge_size: size of spatial merge blocks (default: 2)

    Returns:
        torch.Tensor: Patch positions in 2x2 block order, same shape [t*h*w, 3]
    """
    sms = spatial_merge_size
    if sms == 1:
        return positions

    device = positions.device
    total_patches = t * h * w

    # Generate row-major indices: [0, 1, 2, ..., t*h*w-1]
    # Reshape to [t, h, w]
    indices = torch.arange(total_patches, device=device).view(t, h, w)

    # Calculate merged dimensions
    h_merged = h // sms
    w_merged = w // sms

    # Reshape to [t, h_merged, sms, w_merged, sms]
    indices = indices.view(t, h_merged, sms, w_merged, sms)

    # Permute to [t, h_merged, w_merged, sms_h, sms_w] - 2x2 block order
    indices = indices.permute(0, 1, 3, 2, 4).contiguous()

    # Flatten to get the reordering indices
    indices = indices.view(total_patches)

    # Apply the reordering to positions
    return positions[indices]


class OneVisionEncoderModel(VisionModule):
    """
    OneVision encoder model with packed sequence support, pre-layernorm and 3D RoPE.

    Enhanced vision transformer that supports variable-length sequences packed together
    for efficient batch processing. Uses cumulative sequence lengths to handle multiple
    samples with different numbers of tokens.

    This model uses 3D (T, H, W) rotary position embeddings with 4:6:6 dimension split
    for temporal/spatial encoding, matching the HuggingFace LlavaOnevision2 implementation.

    Args:
        config (VisionConfig): Vision model configuration.
        transformer_layer_spec (ModuleSpec): Specification for transformer layers.
        spatial_merge_size (int): Size for spatial merging. Default: 2.
    """

    def __init__(
        self,
        config: VisionConfig,
        transformer_layer_spec: ModuleSpec,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__(config)
        self.model_type = ModelType.encoder_or_decoder
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        # 3D Rotary position embedding with 4:6:6 split (T:H:W)
        self.video_rope = VideoRotaryEmbeddingSplit466(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            rope_theta=10000.0,  # Default rope_theta for vision
        )

        # Patch embedding module
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        # Transformer decoder blocks
        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
        )

        # Pre-layer normalization applied before transformer blocks
        self.pre_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-4)

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """
        Set input tensor for the model (used in pipeline parallelism).

        Args:
            input_tensor (torch.Tensor): Input tensor from previous pipeline stage.
        """
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self, x: torch.Tensor, grid_thw: torch.Tensor, patch_positions: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        """
        Forward pass with packed sequence support and 3D RoPE.

        Processes batched inputs with variable sequence lengths using packed sequences
        for efficient attention computation. Uses 3D (T, H, W) rotary position embeddings.

        Args:
            x (torch.Tensor): Input patches of shape [total_patches, channels * patch_size * patch_size].
            grid_thw (torch.Tensor): Grid dimensions [batch_size, 3] with (T, H, W) per sample.
                Note: H and W are the UNMERGED patch counts. The actual token count after spatial
                merge is t * (h / spatial_merge_size) * (w / spatial_merge_size) * spatial_merge_size^2 = t * h * w.

        Returns:
            torch.Tensor: Output embeddings of shape [total_patches, hidden_size].
        """
        # Convert patches to embeddings
        x = self.patch_embed(x)

        batch_size = grid_thw.size(0)
        seq_len, hidden_dim = x.size()
        sms = self.spatial_merge_size

        # Generate 3D rotary position embeddings for each sample and concatenate
        # Note: When spatial_merge_size=2, patches are arranged in 2x2 blocks:
        # [p(0,0), p(0,1), p(1,0), p(1,1)], [p(0,2), p(0,3), p(1,2), p(1,3)], ...
        # We first generate RoPE in row-major order, then convert to block order.
        all_rotary_pos_emb = []
        tokens_per_sample = []

        if patch_positions is not None:
            # Use provided patch positions (from pre-computed data)
            # patch_positions is [total_patches, 3] with (t, h, w) per patch
            # Need to reorder from row-major to 2x2 block layout for each sample

            offset = 0
            for i in range(batch_size):
                t, h, w = grid_thw[i]
                t, h, w = t.item(), h.item(), w.item()
                num_patches = t * h * w
                tokens_per_sample.append(num_patches)

                # Extract this sample's positions
                sample_positions = patch_positions[offset : offset + num_patches]

                # Convert positions from row-major to 2x2 block layout
                sample_positions = convert_positions_to_block_layout(sample_positions, t, h, w, sms)

                # Compute RoPE from reordered positions
                sample_freqs = self.video_rope.forward_from_positions(sample_positions)
                all_rotary_pos_emb.append(sample_freqs)

                offset += num_patches
        else:
            # Generate positions from grid_thw (original behavior)
            for i in range(batch_size):
                t, h, w = grid_thw[i]
                t, h, w = t.item(), h.item(), w.item()
                tokens_per_sample.append(t * h * w)

                # Generate RoPE in row-major order (original 1x1 layout)
                sample_freqs = self.video_rope(t=t, h=h, w=w, device=x.device)
                # sample_freqs shape: [t * h * w, half]

                # Convert from row-major (1x1) to 2x2 block layout
                sample_freqs = convert_rope_to_block_layout(sample_freqs, t, h, w, sms)

                all_rotary_pos_emb.append(sample_freqs)

        rotary_pos_emb = torch.cat(all_rotary_pos_emb, dim=0)

        # Build cumulative sequence lengths for packed sequence attention
        cu_seqlens = []
        cumulative_length = 0
        cu_seqlens.append(cumulative_length)

        for length in tokens_per_sample:
            cumulative_length += int(length)
            cu_seqlens.append(cumulative_length)

        cu_seqlens = torch.tensor(
            cu_seqlens,
            device=x.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )

        # Add sequence dimension:  [s, h] -> [s, 1, h]
        x = x[:, None, :].contiguous()

        # Apply pre-layer normalization
        x = self.pre_layernorm(x)

        # Pass through transformer with packed sequence parameters
        x = self.decoder(
            x,
            packed_seq_params=[
                PackedSeqParams(
                    qkv_format="thd",  # (total_tokens, num_heads, head_dim) format
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                )
                for i in range(self.config.num_layers)
            ],
            rotary_pos_emb=rotary_pos_emb.unsqueeze(1).unsqueeze(2),
            attention_mask=None,
            attn_mask_type=AttnMaskType.no_mask,
        )

        # Remove sequence dimension:  [s, 1, h] -> [s, h]
        x = x[:, 0, :].contiguous()

        return x

    def forward_debug(self, x: torch.Tensor, grid_thw: torch.Tensor) -> dict:
        """
        Debug version of forward pass that captures intermediate states.

        Identical to forward() but saves intermediate outputs at key stages
        for debugging and analysis purposes.

        Args:
            x (torch.Tensor): Input patches of shape [total_patches, channels * patch_size * patch_size].
            grid_thw (torch.Tensor): Grid dimensions [batch_size, 3] with (T, H, W) per sample.
                Note: H and W are the UNMERGED patch counts.

        Returns:
            dict: Dictionary containing intermediate outputs:
                - "after_patch_embed": Embeddings after patch projection
                - "rotary_pos_emb":  Rotary position embeddings
                - "cu_seqlens": Cumulative sequence lengths for packed sequences
                - "before_pre_layernorm":  Embeddings before normalization
                - "after_pre_layernorm": Embeddings after normalization
                - "after_decoder": Embeddings after transformer decoder
                - "before_adapter": Final output embeddings (before any adapter layers)
        """
        output = {}

        # Store input for consistency checking
        output["input_pixel_values"] = x.clone()
        output["input_grid_thw"] = grid_thw.clone()

        # Convert patches to embeddings
        x = self.patch_embed(x)
        output["after_patch_embed"] = x.clone()

        batch_size = grid_thw.size(0)
        seq_len, hidden_dim = x.size()
        sms = self.spatial_merge_size

        # Generate 3D rotary position embeddings for each sample and concatenate
        # Note: When spatial_merge_size=2, patches are arranged in 2x2 blocks.
        # We first generate RoPE in row-major order, then convert to block order.
        all_rotary_pos_emb = []
        tokens_per_sample = []
        for i in range(batch_size):
            t, h, w = grid_thw[i]
            t, h, w = t.item(), h.item(), w.item()
            tokens_per_sample.append(t * h * w)

            # Generate RoPE in row-major order (original 1x1 layout)
            sample_freqs = self.video_rope(t=t, h=h, w=w, device=x.device)
            # sample_freqs shape: [t * h * w, half]

            # Convert from row-major (1x1) to 2x2 block layout
            sample_freqs = convert_rope_to_block_layout(sample_freqs, t, h, w, sms)

            all_rotary_pos_emb.append(sample_freqs)

        rotary_pos_emb = torch.cat(all_rotary_pos_emb, dim=0)
        output["rotary_pos_emb"] = rotary_pos_emb.clone()

        # Build cumulative sequence lengths
        cu_seqlens = []
        cumulative_length = 0
        cu_seqlens.append(cumulative_length)

        for length in tokens_per_sample:
            cumulative_length += int(length)
            cu_seqlens.append(cumulative_length)

        cu_seqlens = torch.tensor(
            cu_seqlens,
            device=x.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        output["cu_seqlens"] = cu_seqlens.clone()

        # Add sequence dimension
        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]
        output["before_pre_layernorm"] = x.clone()

        # Apply pre-layer normalization
        x = self.pre_layernorm(x)
        output["after_pre_layernorm"] = x.clone()

        # Pass through decoder with layer-by-layer capture
        # Use forward_debug to capture layer-by-layer outputs
        decoder_debug_output = self.decoder.forward_debug(
            x,
            packed_seq_params=[
                PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                )
                for i in range(self.config.num_layers)
            ],
            rotary_pos_emb=rotary_pos_emb.unsqueeze(1).unsqueeze(2),
            attention_mask=None,
            attn_mask_type=AttnMaskType.no_mask,
        )

        # Extract layer outputs and final output from decoder
        output["layer_outputs"] = decoder_debug_output.get("layer_outputs", {})
        x = decoder_debug_output.get("final_output", decoder_debug_output.get("before_final_layernorm", x))
        output["after_decoder"] = x.clone()

        # Remove sequence dimension
        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]
        output["before_adapter"] = x.clone()

        return output
