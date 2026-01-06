import argparse
import os
from io import BytesIO

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from llavaonevision2.configuration_llava_onevision2 import LlavaOnevision2Config
from llavaonevision2.modeling_llava_onevision2 import LlavaOnevision2ForConditionalGeneration
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    Qwen2Tokenizer,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)
CUDA_DEVICE = 0


def cosine_similarity(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    min_len = min(len(a), len(b))
    a, b = a[:min_len].numpy(), b[:min_len].numpy()
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if norm_a == 0 or norm_b == 0 else float(np.dot(a, b) / (norm_a * norm_b))


def create_test_image():
    img = Image.new("RGB", (560, 560), color="red")
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 474, 474], fill="blue")
    draw.text((100, 100), "TEST", fill="white")
    return img


def load_empty_model(llm_path):
    print("Loading tokenizer and processor from Qwen2.5-VL and empty model...")
    tokenizer = Qwen2Tokenizer.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True, device_map={"": f"cuda:{CUDA_DEVICE}"}, use_fast=True
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
    processor.image_processor.temporal_patch_size = 1
    processor.image_processor.max_pixels = 1600*1600
    llava_ov_config = LlavaOnevision2Config()
    llm_config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True, use_fast=True)
    llava_ov_config.text_config.update(llm_config.to_dict())
    llava_ov_config.text_config.tie_word_embeddings = False
    # Set both text_hidden_size and out_hidden_size for merger to output correct dimension
    llava_ov_config.vision_config.text_hidden_size = llava_ov_config.text_config.hidden_size
    llava_ov_config.vision_config.out_hidden_size = (
        llava_ov_config.text_config.hidden_size
    )  # Critical: merger output dim

    model = LlavaOnevision2ForConditionalGeneration(llava_ov_config)
    return model, processor, tokenizer


def load_vit_weights(model, vit_path):
    """
    Load ViT weights and copy them to the vision part of LLaVAOneVision2_ForConditionalGeneration

    Args:
        model: LLaVAOneVision2_ForConditionalGeneration
        vit_path: ViT model path
    """
    print(f"Loading weight form: {vit_path}")

    if os.path.exists(vit_path):
        print(f"Loading weights from local file: {vit_path}")
        cache_path = os.path.join(vit_path, "model.safetensors")
    else:
        print(f"Loading weights from Hugging Face Hub: {vit_path}")
        cache_path = hf_hub_download(vit_path, "model.safetensors")

    vit_weights = load_file(cache_path)
    loaded_keys = 0

    def convert_state_dict(state_dict):
        """
        Convert ViT state dict with separate q_proj, k_proj, v_proj to merged qkv format.
        Also rename out_proj to proj.
        """
        new_state_dict = {}
        # Collect q, k, v weights for merging
        qkv_weights = {}  # {layer_prefix: {'q': tensor, 'k': tensor, 'v': tensor}}
        qkv_biases = {}  # {layer_prefix: {'q': tensor, 'k': tensor, 'v': tensor}}

        for key, value in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            # Skip head.* keys (classification head, not needed for encoder)
            if key.startswith("head."):
                continue
            # Skip post layernorm keys if present
            if key.startswith("layernorm_post."):
                continue

            # Handle q_proj, k_proj, v_proj -> qkv merging
            if ".self_attn.q_proj." in key or ".self_attn.k_proj." in key or ".self_attn.v_proj." in key:
                # Extract layer prefix (e.g., "encoder.layers.0.self_attn")
                if ".q_proj." in key:
                    layer_prefix = key.replace(".q_proj.weight", "").replace(".q_proj.bias", "")
                    proj_type = "q"
                elif ".k_proj." in key:
                    layer_prefix = key.replace(".k_proj.weight", "").replace(".k_proj.bias", "")
                    proj_type = "k"
                else:  # v_proj
                    layer_prefix = key.replace(".v_proj.weight", "").replace(".v_proj.bias", "")
                    proj_type = "v"

                is_weight = key.endswith(".weight")

                if is_weight:
                    if layer_prefix not in qkv_weights:
                        qkv_weights[layer_prefix] = {}
                    qkv_weights[layer_prefix][proj_type] = value
                else:  # bias
                    if layer_prefix not in qkv_biases:
                        qkv_biases[layer_prefix] = {}
                    qkv_biases[layer_prefix][proj_type] = value
                continue

            # Handle out_proj -> proj renaming
            if ".self_attn.out_proj." in key:
                new_key = key.replace(".out_proj.", ".proj.")
                new_key = "model.visual." + new_key
                new_state_dict[new_key] = value
                continue

            # Simply add model.visual. prefix for other keys
            new_key = "model.visual." + key
            new_state_dict[new_key] = value

        # Merge q, k, v weights into qkv
        for layer_prefix, qkv_dict in qkv_weights.items():
            if "q" in qkv_dict and "k" in qkv_dict and "v" in qkv_dict:
                # Stack in order: q, k, v
                merged_weight = torch.cat([qkv_dict["q"], qkv_dict["k"], qkv_dict["v"]], dim=0)
                new_key = "model.visual." + layer_prefix + ".qkv.weight"
                new_state_dict[new_key] = merged_weight
                print(f"Merged QKV weights for {layer_prefix}: {merged_weight.shape}")

        # Merge q, k, v biases into qkv
        for layer_prefix, qkv_dict in qkv_biases.items():
            if "q" in qkv_dict and "k" in qkv_dict and "v" in qkv_dict:
                # Stack in order: q, k, v
                merged_bias = torch.cat([qkv_dict["q"], qkv_dict["k"], qkv_dict["v"]], dim=0)
                new_key = "model.visual." + layer_prefix + ".qkv.bias"
                new_state_dict[new_key] = merged_bias

        return new_state_dict

    vit_weights = convert_state_dict(vit_weights)
    vit_keys = len(set(vit_weights.keys()))

    model_state_dict = model.state_dict()
    total_keys = len(model_state_dict.keys())
    for vit_key in vit_weights:
        if vit_key not in model_state_dict:
            logger.warning(f"ViT key {vit_key} not found in model, skipping...")
            continue
        model_state_dict[vit_key] = vit_weights[vit_key].clone()
        loaded_keys += 1
    assert loaded_keys == vit_keys, f"ViT weight loading incomplete: {loaded_keys}/{vit_keys} parameters loaded"
    model.load_state_dict(model_state_dict)
    print(f"ViT weights loaded successfully: {loaded_keys}/{total_keys} parameters loaded")

    return vit_weights, loaded_keys


def load_adapter_weights(model, adapter_path, cur_len):
    """
    Load Adapter weights and copy them to the corresponding part of LLaVAOneVision2_ForConditionalGeneration

    Args:
        model: LLaVAOneVision2_ForConditionalGeneration model
        adapter_path: Adapter model path
    """
    print(f"Loading Adapter weights from: {adapter_path}")

    # Load Adapter weights
    if adapter_path.endswith(".safetensors"):
        adapter_weights = load_file(adapter_path)
    else:
        adapter_weights = torch.load(adapter_path, map_location="cpu")
        if "state_dict" in adapter_weights:
            adapter_weights = adapter_weights["state_dict"]

    # Count successfully loaded parameters
    loaded_keys = 0
    total_keys = 0
    ADAPTER_KEYS_TO_MODIFY_MAPPING = {"model.mm_projector": "model.visual.merger"}

    def convert_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            for key_to_modify, new_key in ADAPTER_KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in key:
                    key = key.replace(key_to_modify, new_key)

            new_state_dict[key] = value
        return new_state_dict

    adapter_weights = convert_state_dict(adapter_weights)
    adapter_keys = len(set(adapter_weights.keys()))

    # Load weights into model
    model_state_dict = model.state_dict()
    total_keys = len(model_state_dict.keys())
    for adapter_key in adapter_weights:
        if adapter_key not in model_state_dict:
            logger.warning(f"Adapter key {adapter_key} not found in model, skipping...")
            continue
        model_state_dict[adapter_key] = adapter_weights[adapter_key].clone()
        loaded_keys += 1
    assert loaded_keys == adapter_keys, (
        f"Adapter weight loading incomplete: {loaded_keys}/{adapter_keys} parameters loaded"
    )
    model.load_state_dict(model_state_dict)
    print(f"Adapter weights loaded successfully: {loaded_keys + cur_len}/{total_keys} parameters loaded")

    return adapter_weights, cur_len + loaded_keys


def load_llm_weights(model, llm_path, cur_len):
    """
    Args:
        model: LLaVAOneVision2_ForConditionalGeneration model
        llm_path: LLM model path
    """
    print(f"Loading weight form: {llm_path}")
    if os.path.exists(llm_path):
        cache_path = llm_path
    else:
        cache_path = snapshot_download(llm_path, allow_patterns="*.safetensors")

    llm_weights = {}
    if os.path.isdir(cache_path):
        for filename in os.listdir(cache_path):
            if filename.endswith(".safetensors"):
                filepath = os.path.join(cache_path, filename)
                weights = load_file(filepath)
                llm_weights.update(weights)
    elif cache_path.endswith(".safetensors"):
        llm_weights = load_file(cache_path)
    else:
        llm_weights = torch.load(cache_path, map_location="cpu")
        if "state_dict" in llm_weights:
            llm_weights = llm_weights["state_dict"]

    loaded_keys = 0

    ADAPTER_KEYS_TO_MODIFY_MAPPING = {
        "model.": "model.language_model.",
    }

    def convert_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            for key_to_modify, new_key in ADAPTER_KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in key:
                    key = key.replace(key_to_modify, new_key)

            new_state_dict[key] = value
        return new_state_dict

    llm_weights = convert_state_dict(llm_weights)
    if "lm_head.weight" not in llm_weights:
        llm_weights["lm_head.weight"] = llm_weights["model.language_model.embed_tokens.weight"]
    llm_keys = len(set(llm_weights.keys()))

    model_state_dict = model.state_dict()
    for llm_key in llm_weights:
        if llm_key not in model_state_dict:
            logger.warning(f"LLM key {llm_key} not found in model, skipping...")
            continue
        model_state_dict[llm_key] = llm_weights[llm_key].clone()
        loaded_keys += 1
    assert loaded_keys == llm_keys, f"LLM weight loading incomplete: {loaded_keys}/{llm_keys} parameters loaded"

    return llm_weights


def convert_rowmajor_to_block_layout(features: torch.Tensor, t: int, h: int, w: int, spatial_merge_size: int = 2) -> torch.Tensor:
    """
    Convert features from row-major order to 2x2 block layout.

    Row-major order: patches are ordered as [p(0,0), p(0,1), p(0,2), ..., p(1,0), p(1,1), ...]
    Block order: patches are grouped in 2x2 blocks: [p(0,0), p(0,1), p(1,0), p(1,1)], [p(0,2), p(0,3), p(1,2), p(1,3)], ...

    Args:
        features: Feature tensor in row-major order, shape [seq_len, hidden_dim]
        t: temporal dimension (number of frames)
        h: height (number of vertical patches)
        w: width (number of horizontal patches)
        spatial_merge_size: size of spatial merge blocks (default: 2)

    Returns:
        torch.Tensor: Features in 2x2 block order, same shape [seq_len, hidden_dim]
    """
    sms = spatial_merge_size
    if sms == 1:
        return features

    hidden_dim = features.shape[-1]

    # features shape: [t*h*w, hidden_dim]
    # Reshape to [t, h, w, hidden_dim]
    features = features.view(t, h, w, hidden_dim)

    # Calculate merged dimensions
    h_merged = h // sms
    w_merged = w // sms

    # Reshape to [t, h_merged, sms, w_merged, sms, hidden_dim]
    features = features.view(t, h_merged, sms, w_merged, sms, hidden_dim)

    # Permute to [t, h_merged, w_merged, sms_h, sms_w, hidden_dim] - 2x2 block order
    features = features.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Reshape back to [t*h*w, hidden_dim]
    features = features.view(t * h * w, hidden_dim)

    return features


def convert_block_to_rowmajor_layout(features: torch.Tensor, t: int, h: int, w: int, spatial_merge_size: int = 2) -> torch.Tensor:
    """
    Convert features from 2x2 block layout back to row-major order.
    This is the inverse of convert_rowmajor_to_block_layout.

    Block order: patches are grouped in 2x2 blocks: [p(0,0), p(0,1), p(1,0), p(1,1)], [p(0,2), p(0,3), p(1,2), p(1,3)], ...
    Row-major order: patches are ordered as [p(0,0), p(0,1), p(0,2), ..., p(1,0), p(1,1), ...]

    Args:
        features: Feature tensor in block order, shape [seq_len, hidden_dim]
        t: temporal dimension (number of frames)
        h: height (number of vertical patches)
        w: width (number of horizontal patches)
        spatial_merge_size: size of spatial merge blocks (default: 2)

    Returns:
        torch.Tensor: Features in row-major order, same shape [seq_len, hidden_dim]
    """
    sms = spatial_merge_size
    if sms == 1:
        return features

    hidden_dim = features.shape[-1]

    # Calculate merged dimensions
    h_merged = h // sms
    w_merged = w // sms

    # features shape: [t*h*w, hidden_dim]
    # Reshape to [t, h_merged, w_merged, sms_h, sms_w, hidden_dim]
    features = features.view(t, h_merged, w_merged, sms, sms, hidden_dim)

    # Permute back to [t, h_merged, sms_h, w_merged, sms_w, hidden_dim]
    features = features.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Reshape to [t, h, w, hidden_dim] and then [t*h*w, hidden_dim]
    features = features.view(t * h * w, hidden_dim)

    return features


def validate_vit_consistency(model, vit_path, img_path, processor):
    """
    Verify the consistency of the ViT component.

    The original ViT uses CLIP image processor (row-major patch order),
    while the merged model uses Qwen2VL processor (2x2 block patch order).

    Since both processors use the same mean/std normalization, the pixel values
    are identical - only the patch arrangement differs. We convert the original
    ViT's output from row-major to block order for comparison.

    Args:
        model: LlavaOnevision2ForConditionalGeneration after merged
        vit_path: original ViT model path
        img_path: sample image path/url
        processor: Processor containing CLIPImageProcessor for original ViT
    """
    print("Verifying consistency of ViT component...")

    from onevision_encoder import OneVisionEncoderModel
    from qwen_vl_utils import process_vision_info

    device = torch.device(f"cuda:{CUDA_DEVICE}")

    # Load image
    if img_path.startswith("http"):
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(img_path).convert("RGB")

    patch_size = 14  # ViT patch size
    merge_size = 2  # spatial merge size

    pixel_unit = patch_size * merge_size  # 28
    orig_w, orig_h = image.size
    new_h = (orig_h // pixel_unit) * pixel_unit
    new_w = (orig_w // pixel_unit) * pixel_unit
    if new_h != orig_h or new_w != orig_w:
        image = image.resize((new_w, new_h), Image.BILINEAR)
        print(f"Resized image from ({orig_w}, {orig_h}) to ({new_w}, {new_h}) for patch alignment")

    # ========== Original ViT with CLIP processor (row-major order) ==========
    clip_image_processor = CLIPImageProcessor.from_pretrained(vit_path, local_files_only=True)
    clip_pixel_values = clip_image_processor(
        images=image, return_tensors="pt", do_resize=False, do_center_crop=False
    )["pixel_values"]
    clip_pixel_values = clip_pixel_values.to(dtype=torch.bfloat16, device=device)

    _, _, H, W = clip_pixel_values.shape
    grid_h = H // patch_size
    grid_w = W // patch_size
    print(f"Image size: {H}x{W}, grid: [1, {grid_h}, {grid_w}]")

    # ========== Merged model with Qwen2VL processor (2x2 block order) ==========
    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
    # Disable resizing to ensure same image dimensions
    qwen_processor.image_processor.do_resize = False
    qwen_processor.image_processor.max_pixels = H * W
    qwen_processor.image_processor.min_pixels = H * W
    qwen_processor.image_processor.temporal_patch_size = 1

    # Prepare Qwen2VL-style input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Process with Qwen2VL processor
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    qwen_inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    qwen_pixel_values = qwen_inputs["pixel_values"].to(dtype=torch.bfloat16, device=device)
    qwen_grid_thw = qwen_inputs["image_grid_thw"].to(device=device)

    qwen_t, qwen_h, qwen_w = qwen_grid_thw[0].tolist()
    print(f"Qwen2VL pixel_values shape: {qwen_pixel_values.shape}, grid_thw: [{qwen_t}, {qwen_h}, {qwen_w}]")

    # Verify grid dimensions match
    assert (1, grid_h, grid_w) == (qwen_t, qwen_h, qwen_w), (
        f"Grid dimension mismatch! CLIP: [1, {grid_h}, {grid_w}], Qwen2VL: [{qwen_t}, {qwen_h}, {qwen_w}]"
    )

    # ========== Load models ==========
    merged_visual = model.model.visual.to(dtype=torch.bfloat16, device=device)
    merged_visual.eval()

    # Load original ViT with flash_attention_2
    # First load as float32, then convert to bfloat16 (same as merged_visual)
    original_vit = OneVisionEncoderModel.from_pretrained(
        vit_path, trust_remote_code=True, attn_implementation="flash_attention_2"
    ).to(dtype=torch.bfloat16, device=device)
    original_vit.eval()

    # ========== Forward pass ==========
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Original ViT with CLIP-processed input (row-major order output)
            original_output = original_vit(clip_pixel_values, use_head=False).last_hidden_state[0]

            # Merged model with Qwen2VL-processed input (block order output)
            merged_output = merged_visual(qwen_pixel_values, qwen_grid_thw, skip_merger=True).last_hidden_state[0]

            print(f"Original ViT output shape: {original_output.shape}")
            print(f"Merged visual output shape: {merged_output.shape}")

            # Convert original ViT output from row-major to block layout for comparison
            # Both processors use same mean/std, so pixel values are identical,
            # only the patch arrangement differs
            t, h, w = 1, grid_h, grid_w
            original_output_block = convert_rowmajor_to_block_layout(
                original_output, t, h, w, spatial_merge_size=merge_size
            )

            # Compare outputs
            cur_sim = cosine_similarity(merged_output.flatten().cpu(), original_output_block.flatten().cpu())

        merged_output = merged_output.float()
        original_output_block = original_output_block.float()
        diff = (merged_output - original_output_block).abs().mean().item()

    print(f"ViT output mean difference: {diff:.8f}")
    print(f"ViT output cosine similarity: {cur_sim:.6f}")

    if diff < 1e-1 and cur_sim > 0.99:
        print("✅ ViT component consistency verification passed")
    else:
        raise ValueError("❌ ViT component consistency verification failed")

    model.model.visual = merged_visual.to("cpu")
    del original_vit
    torch.cuda.empty_cache()


def validate_llm_consistency(model, llm_path, sample_text):
    """
    Verify the consistency of the LLM component

    Args:
        model: Merged LLaVAOneVision2_ForConditionalGeneration model
        llm_path: Original LLM model path
        sample_text: Sample text
    """
    print("Verifying consistency of LLM component...")

    # Get dtype and device from language_model, not from visual
    # (visual may have different dtype after validate_vit_consistency)
    device = next(model.model.language_model.parameters()).device
    dtype = next(model.model.language_model.parameters()).dtype

    # Load original LLM model
    original_llm = AutoModelForCausalLM.from_pretrained(llm_path).to(dtype=dtype, device=device)
    original_llm.eval()
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)

    # Prepare sample text
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)

    with torch.no_grad():
        # Use model.model.language_model directly instead of full model forward
        merged_lm = model.model.language_model
        merged_lm_head = model.lm_head

        # Pass the same inputs to both models for fair comparison
        merged_output = merged_lm(
            input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask", None)
        ).last_hidden_state
        merged_output = merged_lm_head(merged_output)

        # Original LLM output
        original_output = original_llm(**inputs).logits

        cur_sim = cosine_similarity(merged_output.flatten(0, 1).cpu(), original_output.flatten(0, 1).cpu())

    # Compare results
    diff = (merged_output - original_output).abs().mean().item()
    max_diff = (merged_output - original_output).abs().max().item()
    print(f"LLM output mean difference: {diff:.8f}")
    print(f"LLM output max difference: {max_diff:.8f}")
    print(f"LLM output cosine similarity: {cur_sim:.6f}")

    # Cosine similarity > 0.99 is sufficient to verify consistency
    # Mean difference threshold is relaxed because logits can have large absolute values
    if cur_sim > 0.99 and diff < 1e-2:
        print("✅ LLM component consistency verification passed")
    else:
        raise ValueError("❌ LLM component consistency verification failed")

    # Cleanup
    del original_llm
    torch.cuda.empty_cache()


def save_merged_model(model, output_path, tokenizer, image_processor):
    """
    Save the merged model

    Args:
        model: Merged model
        output_path: Output path
    """
    print(f"Saving merged model to: {output_path}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save model configuration
    tokenizer.save_pretrained(output_path)
    image_processor.save_pretrained(output_path)
    model.save_pretrained(output_path)

    print("Model saving completed.")


def validate_end_to_end(model, processor, tokenizer, img_path):
    """
    Validate that the merged model can process an image+text sample end-to-end.
    This ensures the full pipeline (image encoding -> projection -> LLM generation) works correctly.

    Args:
        model: Merged LlavaOnevision2ForConditionalGeneration model
        processor: Processor with image processor
        tokenizer: Tokenizer for text processing
        img_path: Path or URL to test image
    """
    print("=" * 60)
    print("Validating end-to-end inference with image+text sample...")
    print("=" * 60)

    from qwen_vl_utils import process_vision_info

    device = torch.device(f"cuda:{CUDA_DEVICE}")

    # Load image
    if img_path.startswith("http"):
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(img_path).convert("RGB")

    print(f"Loaded image: {image.size}")

    # Move model to device with bfloat16
    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()

    # Prepare Qwen2VL-style input with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]

    # Process with Qwen2VL processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    print(f"Image grid THW: {inputs['image_grid_thw']}")

    # Generate output
    print("Running forward pass and generation...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # First test forward pass (without generation) to check gradients flow
            try:
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                )
                print(f"Forward pass successful! Logits shape: {outputs.logits.shape}")
            except Exception as e:
                raise ValueError(f"❌ Forward pass failed: {e}")

            # Then test generation
            try:
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                    max_new_tokens=50,
                    do_sample=False,  # Greedy decoding for reproducibility
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                print(f"Generation successful! Generated IDs shape: {generated_ids.shape}")
            except Exception as e:
                raise ValueError(f"❌ Generation failed: {e}")

    # Decode the generated text
    # Only decode the newly generated tokens (exclude input tokens)
    input_len = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)

    print("-" * 40)
    print("Generated text:")
    print(generated_text)
    print("-" * 40)

    # Basic sanity check: generated text should not be empty
    if len(generated_text.strip()) == 0:
        print("⚠️ Warning: Generated text is empty (model may not be trained yet)")
    else:
        print(f"✅ End-to-end validation passed! Generated {len(generated_text)} characters.")

    # Move model back to CPU to free GPU memory
    model = model.to("cpu")
    torch.cuda.empty_cache()

    return generated_text


def main(args):
    # model paths
    vit_path = args.vit_path
    adapter_path = args.adapter_path
    llm_path = args.llm_path
    output_path = args.output_path
    img_path = args.img_path
    sample_text = args.sample_text

    # 1. load empty model
    model, processor, tokenizer = load_empty_model(llm_path)
    model.to(dtype=torch.float32)

    print("Processor:", processor)

    pretrain_weights = {}
    # 2. load ViT weights
    vit_weights, cur_len = load_vit_weights(model, vit_path)
    pretrain_weights.update(vit_weights)

    # 3. load Adapter weights
    if adapter_path:
        adapter_weights, cur_len = load_adapter_weights(model, adapter_path, cur_len)
        pretrain_weights.update(adapter_weights)

    # 4. load LLM weights
    llm_weights = load_llm_weights(model, llm_path, cur_len)
    pretrain_weights.update(llm_weights)

    model.load_state_dict(pretrain_weights, strict=False)

    # 5. validate model consistency
    validate_vit_consistency(model, vit_path, img_path, processor)
    validate_llm_consistency(model, llm_path, sample_text)

    # 6. validate end-to-end inference with image+text
    validate_end_to_end(model, processor, tokenizer, img_path)

    # 7. save merged model
    save_merged_model(model.to(dtype=torch.bfloat16), output_path, tokenizer, processor)
    print("Model merging process completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ViT and LLM models")
    parser.add_argument(
        "--vit_path", type=str, default="lmms-lab-encoder/onevision-encoder-large", help="Path to the ViT model"
    )
    parser.add_argument("--llm_path", type=str, default="Qwen3-1.7B-Base/", help="Path to the LLM model")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./checkpoints/merged/LLaVA-OneVision-2-2B-stage0",
        help="Path to save the merged model",
    )
    parser.add_argument("--adapter_path", type=str, default="", help="Path to the Adapter model (optional)")
    parser.add_argument(
        "--img_path",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="Path to the image file",
    )
    parser.add_argument(
        "--sample_text", type=str, default="Hello, my dog is cute", help="Sample text for LLM consistency check"
    )
    args = parser.parse_args()
    main(args)
