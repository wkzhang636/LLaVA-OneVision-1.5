"""Qwen2VLTaskEncoder class."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, TypeVar, Union

import numpy as np
import torch
from megatron.energon import CaptioningSample, VQASample
from megatron.energon.errors import SampleException
from megatron.energon.flavors.base_dataset import (
    BaseCoreDatasetFactory,
    SavableDataset,
)
from megatron.energon.flavors.crude import CrudeWebdataset
from megatron.energon.flavors.webdataset import VideoData
from megatron.energon.metadataset.loader_interface import DatasetBlendMode
from megatron.energon.task_encoder.base import stateless
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers import (
    BlendDataset,
    EpochizeDataset,
    LogSampleDataset,
    ShuffleBufferDataset,
)
from megatron.energon.wrappers.repeat_dataset import RepeatDataset
from qwen_vl_utils.vision_process import smart_nframes, smart_resize
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from typing_extensions import override

from aiak_training_llm.data.multimodal import MultiMixQASample
from aiak_training_llm.data.multimodal.length_sort_dataset import LengthPoolSortDataset
from aiak_training_llm.utils import constants, get_chat_template
from transformers import AutoProcessor

from .task_encoder import ImageTaskBatchPacked, ImageTaskSample, ImageTaskSamplePacked, TaskEncoder


T = TypeVar("T")
V = TypeVar("V")
T_sample = TypeVar("T_sample")
T_encoded_sample = TypeVar("T_encoded_sample")
T_raw_batch = TypeVar("T_raw_batch")
T_batch = TypeVar("T_batch")


IGNORE_INDEX = -100  # ID for labels that should be ignored.
IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
VISION_TAGS = ["<|vision_start|>", "<|vision_end|>"]
IMAGE_TOKEN_WITH_TAGS = VISION_TAGS[0] + IMAGE_TOKEN + VISION_TAGS[1]
VIDEO_TOKEN_WITH_TAGS = VISION_TAGS[0] + VIDEO_TOKEN + VISION_TAGS[1]


def get_stateless(fn: Callable[..., T_sample]) -> bool:
    """Get whether a function is stateless."""
    return getattr(fn, "__stateless__", False)


@dataclass
class Qwen2VLImageTaskSample(ImageTaskSample):
    """An image task sample with a grid of tokens and their corresponding pixel values."""

    image_grid_thw: torch.Tensor = None
    video_grid_thw: torch.Tensor = None

    def __init__(self, image_grid_thw: str, video_grid_thw=None, **kwargs):
        super().__init__(**kwargs)
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


@dataclass
class Qwen2VLImageTaskSamplePacked(ImageTaskSamplePacked):
    """An image task sample with a grid of tokens and their corresponding pixel values."""

    image_grid_thw: torch.Tensor = None
    video_grid_thw: torch.Tensor = None

    def __init__(self, sample: ImageTaskSample, image_grid_thw: str, video_grid_thw=None):
        super().__init__(**vars(sample))
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


@dataclass
class Qwen2VLImageTaskBatchPacked(ImageTaskBatchPacked):
    """An image task sample with a grid of tokens and their corresponding pixel values."""

    image_grid_thw: torch.Tensor = None
    video_grid_thw: torch.Tensor = None

    def __init__(self, sample: ImageTaskSample, image_grid_thw: str, video_grid_thw=None):
        super().__init__(**vars(sample))
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw


class Qwen2VLTaskEncoder(TaskEncoder):
    """A simple task encoder for VLMs."""

    def __init__(self, args):
        super().__init__()
        if args.training_phase in ["sft"]:
            self.chat_template = get_chat_template()
        self.processor = AutoProcessor.from_pretrained(self.args.hf_tokenizer_path, trust_remote_code=True)

        # image
        self.min_pixels = args.min_pixels
        self.max_pixels = args.max_pixels

    def _resize_image(self, image, size_factor=28):
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=size_factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        image = image.resize((resized_width, resized_height))

        return image

    def _process(self, image, text):
        """ " Process the data to get the model's input"""
        inputs = self.processor(
            text=text,
            images=image,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"][0]
        attn_mask = inputs["attention_mask"][0].logical_not()
        image_grid_thw = None
        pixel = []
        if image is not None:
            image_grid_thw = inputs["image_grid_thw"]  # [t,h,w]
            pixel = [inputs["pixel_values"]]  # [hw, 2*3*14*14]

        target = input_ids.clone()
        vision_start_id, img_pad_id, vision_end_id = self.tokenizer.convert_tokens_to_ids(
            [VISION_TAGS[0], IMAGE_TOKEN, VISION_TAGS[1]]
        )
        target[target == vision_start_id] = IGNORE_INDEX
        target[target == img_pad_id] = IGNORE_INDEX
        target[target == vision_end_id] = IGNORE_INDEX

        return input_ids, target, pixel, image_grid_thw, attn_mask

    def process_sft_vqa(self, context, answer, image):
        """process the data for sft vqa"""
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": context}, {"role": "assistant", "content": answer}], tokenize=False
        ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)
        if text[-1] == "\n":
            text = text[:-1]
        input_ids, _, imgs, image_grid_thw, attn_mask = self._process(image, text)
        target = torch.ones_like(input_ids) * IGNORE_INDEX
        answer = self.tokenizer.tokenize(answer)
        target[-len(answer) - 1 : -1] = torch.tensor(answer)

        return input_ids, target, attn_mask, imgs, image_grid_thw

    def compute_frame_timestamps(self, images, patch_positions, fps):
        """Compute timestamps for unique frames from patch positions.

        Args:
            images: List of images
            patch_positions: Tensor of shape (n, 3) with (t, h, w) coordinates,
                             where t is the frame index
            fps: Frames per second

        Returns:
            List of float timestamps for each unique frame
        """
        if patch_positions is None or len(patch_positions) == 0:
            return []

        t_values = patch_positions[:, 0]
        unique_t = torch.unique(t_values)
        timestamps = [float(t.item() / fps) for t in unique_t]

        return timestamps

    def _prepare_messages_with_timestamp(self, messages: list, timestamp: list):
        """Insert timestamp into messages according to fps and patch positions.

        Args:
            messages: List of message dicts with 'content' field
            timestamp: List of timestamps for each image

        Returns:
            List of messages with timestamps inserted before <image> tags
        """
        new_messages = []
        for message in messages:
            content = message['content']
            image_count = content.count('<image>')
            if image_count > 0:
                parts = content.split('<image>')
                new_parts = []
                timestamp_idx = 0
                for i, part in enumerate(parts):
                    new_parts.append(part)
                    if i < len(parts) - 1:
                        if timestamp_idx < len(timestamp):
                            new_parts.append(f"<{timestamp[timestamp_idx]:.1f} seconds><image>")
                            timestamp_idx += 1
                        else:
                            new_parts.append("<image>")
                new_content = ''.join(new_parts)
            else:
                new_content = content
            new_messages.append({**message, 'content': new_content})
        return new_messages

    def _insert_timestamp_tokens(
        self,
        input_ids: torch.Tensor,
        target: torch.Tensor,
        attn_mask: torch.Tensor,
        patch_positions: list[torch.Tensor],
        timestamp_tokens: list[list],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Insert timestamp tokens into input_ids based on patch positions.

        Args:
            input_ids: Input token IDs [seq_len]
            target: Target labels [seq_len]
            attn_mask: Attention mask [seq_len]
            patch_positions: List of tensors, each with shape [n_patches, 3] where columns are (t, h, w)
            timestamp_tokens: List of token lists for each timestamp

        Returns:
            Tuple of (new_input_ids, new_target, new_attn_mask)
        """
        if len(timestamp_tokens) == 0 or len(patch_positions) == 0:
            return input_ids, target, attn_mask

        # Flatten patch positions
        flat_patch_positions = torch.cat(patch_positions, dim=0)
        t_values = flat_patch_positions[:, 0]

        # Find positions where t dimension changes
        t_change_patch_indices = []
        prev_t = None
        for i, t in enumerate(t_values):
            if prev_t is None or t != prev_t:
                t_change_patch_indices.append(i)
            prev_t = t

        # Convert patch position indices to image token indices (divide by 4)
        image_token_indices = [patch_idx // 4 for patch_idx in t_change_patch_indices]

        # Determine image token positions in input_ids
        vision_start_id, img_pad_id, vision_end_id = self.tokenizer.convert_tokens_to_ids(
            [VISION_TAGS[0], IMAGE_TOKEN, VISION_TAGS[1]]
        )

        # Find image token positions
        image_token_positions = []
        for pos in range(len(input_ids)):
            if input_ids[pos] == img_pad_id:
                image_token_positions.append(pos)

        # Make sure we have enough image tokens
        if len(image_token_indices) > len(image_token_positions):
            raise ValueError(
                f"Number of timestamps ({len(timestamp_tokens)}) exceeds number of image tokens ({len(image_token_positions)})"
            )

        # Sort in descending order to insert from back to front (to preserve positions)
        insert_indices = list(zip(image_token_indices, timestamp_tokens))
        insert_indices.sort(key=lambda x: x[0], reverse=True)

        # Insert timestamp tokens from back to front
        new_input_ids = input_ids.clone()
        new_target = target.clone()
        new_attn_mask = attn_mask.clone()

        for image_token_idx, ts_tokens in insert_indices:
            if image_token_idx < len(image_token_positions):
                token_pos = image_token_positions[image_token_idx]
                ts_tensor = torch.tensor(ts_tokens, dtype=torch.long)

                # Insert timestamp tokens before this image token
                new_input_ids = torch.cat([
                    new_input_ids[:token_pos],
                    ts_tensor,
                    new_input_ids[token_pos:]
                ], dim=0)

                # Insert IGNORE_INDEX for labels (timestamp tokens should be ignored)
                ignore_tensor = torch.full((len(ts_tokens),), IGNORE_INDEX, dtype=new_target.dtype)
                new_target = torch.cat([
                    new_target[:token_pos],
                    ignore_tensor,
                    new_target[token_pos:]
                ], dim=0)

                # Insert False for attention mask (timestamp tokens are valid tokens)
                attn_tensor = torch.zeros((len(ts_tokens),), dtype=new_attn_mask.dtype)
                new_attn_mask = torch.cat([
                    new_attn_mask[:token_pos],
                    attn_tensor,
                    new_attn_mask[token_pos:]
                ], dim=0)

        return new_input_ids, new_target, new_attn_mask

    def process_sft_qa(self, messages: list, system: str, raw_video: list, raw_image: list, raw_patch_positions: list, **kwargs):
        """process the data for sft qa"""
        video_grid_thw = None
        pixel_values_videos = []
        image_grid_thw = None
        pixel_values_images = []
        video = []
        image = []
        patch_positions = []
        timestamp_tokens = None

        has_image_inputs = raw_image is not None and len(raw_image) > 0
        if has_image_inputs:
            image = raw_image

        if raw_patch_positions is not None:
            for i in raw_patch_positions:
                if i is not None:
                    patch_positions.append(torch.tensor(i, dtype=torch.int64))

        if kwargs is not None and "fps" in kwargs and len(patch_positions) > 0:
            fps = kwargs['fps'][0] if isinstance(kwargs['fps'], list) else kwargs['fps']
            pt_patch_position = torch.concat(patch_positions)
            timestamp = self.compute_frame_timestamps(patch_positions, pt_patch_position, fps)
            timestamp = [round(t, 1) for t in timestamp]
            # If the timestamp is larger than the raw image pair, than means this is for codec data
            # Handle this later in the model forward because we can't insert it here
            if len(timestamp) == len(raw_image):
                messages = self._prepare_messages_with_timestamp(messages, timestamp)
            else:
                timestamp_tokens = []
                for time in timestamp:
                    time_token = self.processor.tokenizer.encode(f"<{time:.1f} seconds>")
                    timestamp_tokens.append(time_token)

        messages, mm_inputs = self.chat_template.mm_plugin.process_messages(
            messages,
            image,
            video if raw_video is not None else [],
            self.processor,
        )

        if has_image_inputs:
            image_grid_thw = mm_inputs.get("image_grid_thw")
            if image_grid_thw is None:
                raise SampleException(
                    "Missing `image_grid_thw` in multimodal inputs:\n"
                    f"- num_raw_image: {len(raw_image)}\n"
                    f"- mm_input_keys: {list(mm_inputs.keys())}\n"
                    f"- has_image_placeholder: {any('<image>' in msg.get('content', '') for msg in messages)}"
                )
            # get image patch num:
            num_patches = 0
            for img_idx in range(len(image_grid_thw)):
                t_val, h_val, w_val = image_grid_thw[img_idx].tolist()
                num_patches += h_val * w_val
            pixel_values = mm_inputs.get("pixel_values")
            if pixel_values is None:
                raise SampleException(
                    "Missing `pixel_values` in multimodal inputs:\n"
                    f"- num_raw_image: {len(raw_image)}\n"
                    f"- mm_input_keys: {list(mm_inputs.keys())}"
                )
            pixel_values_images = [pixel_values]
            if len(patch_positions) == 0:
                # Generate default patch_positions from image_grid_thw, with t=0 for all patches
                # image_grid_thw: [num_images, 3], each row is (t, h, w)
                for img_idx in range(len(image_grid_thw)):
                    t_val, h_val, w_val = image_grid_thw[img_idx].tolist()
                    cur_num_patches = t_val * h_val * w_val
                    # Generate (t, h, w) coordinates in row-major order, t is fixed to 0
                    h_coords = torch.arange(h_val, dtype=torch.int64).repeat_interleave(w_val).repeat(t_val)
                    w_coords = torch.arange(w_val, dtype=torch.int64).repeat(h_val).repeat(t_val)
                    t_coords = torch.zeros(cur_num_patches, dtype=torch.int64)
                    # Stack into [num_patches, 3] tensor
                    img_patch_positions = torch.stack([t_coords, h_coords, w_coords], dim=1)
                    patch_positions.append(img_patch_positions)
            else:
                image_grid_thw = torch.tensor([[len(image_grid_thw),image_grid_thw[0][1],image_grid_thw[0][2]]])
            patch_positions_sum = sum(len(p) for p in patch_positions)
            if num_patches != patch_positions_sum:
                raise SampleException(
                    "num_patches mismatch:\n"
                    f"- num_patches: {num_patches}\n"
                    f"- patch_positions_sum: {patch_positions_sum}\n"
                    f"- len(image_grid_thw): {len(image_grid_thw)}\n"
                    f"- patch_positions len: {len(patch_positions)}\n"
                    f"- image_grid_thw[-1]: {image_grid_thw[-1]}\n"
                    f"- image_sizes: {[img.size for img in image]}"
                )

        encode_pairs = self.chat_template.encode_multiturn(
            tokenizer=self.tokenizer,
            messages=messages,
            system=system,
        )
        input_ids, target = [], []
        for turn_idx, (source_ids, target_ids) in enumerate(encode_pairs):
            input_ids += source_ids + target_ids
            target += [IGNORE_INDEX] * len(source_ids) + target_ids
        input_ids = torch.tensor(input_ids)
        target = torch.tensor(target)
        attn_mask = torch.zeros_like(input_ids).bool()

        # Insert timestamp tokens into input_ids and target
        if timestamp_tokens is not None and len(timestamp_tokens) > 0 and len(patch_positions) > 0:
            input_ids, target, attn_mask = self._insert_timestamp_tokens(
                input_ids=input_ids,
                target=target,
                attn_mask=attn_mask,
                patch_positions=patch_positions,
                timestamp_tokens=timestamp_tokens,
            )

        return (
            input_ids,
            target,
            attn_mask,
            pixel_values_images,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            patch_positions,
        )

    def encode_vqa4packing(self, sample: VQASample) -> ImageTaskSample:
        """Encode VQASample in Qwen2VL style."""
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": sample.context}, {"role": "assistant", "content": sample.answers}],
            tokenize=False,
        ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)

        if text[-1] == "\n":
            text = text[:-1]
            pass

        input_ids, _, imgs, image_grid_thw, attn_mask = self._process(sample.image, text)
        target = torch.ones_like(input_ids) * IGNORE_INDEX
        answers = self.tokenizer.tokenize(sample.answers)
        target[-len(answers) - 1 : -1] = torch.tensor(answers)
        target[-1] = input_ids[-1]
        # print(target[-1])

        num_tiles = [len(image_grid_thw)]
        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert image_grid_thw.prod() / 4 <= self.args.seq_length, f"{sample.__key__} grid_thw: {image_grid_thw}"

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    def encode_multi_vid_qa(self, sample: VQASample) -> ImageTaskSample:
        """Encode sample in Qwen2VL style."""
        if self.args.training_phase == constants.TrainingPhase.SFT:
            input_ids, target, attn_mask, imgs, image_grid_thw, video, video_grid_thw = self.process_sft_qa(
                sample.messages, sample.system, sample.video, None
            )
        else:
            raise NotImplementedError(f"Unknown training phase {self.args.training_phase}")

        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert video_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length, (
                f"{sample.__key__} grid_thw: {video_grid_thw}"
            )

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=video,
            video_grid_thw=video_grid_thw,
            num_tiles=[len(video_grid_thw)],
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    def encode_multi_mix_qa(self, sample: MultiMixQASample) -> ImageTaskSample:
        """Encode sample in Qwen2VL style."""
        if sample.fps is not None:
            # pass
            print(f"Sample key: {sample.__key__}, FPS: {sample.fps}")

        if self.args.training_phase == constants.TrainingPhase.SFT:
            num_tiles = []
            kwargs = {}
            if hasattr(sample, "fps"):
                kwargs['fps'] = sample.fps

            (
                input_ids,
                target,
                attn_mask,
                imgs,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
                patch_positions,
            ) = self.process_sft_qa(sample.messages, sample.system, sample.video, sample.image, sample.patch_positions, **kwargs)
            if sample.video is not None:
                num_tiles = [len(video_grid_thw)]
            elif sample.image is not None:
                num_tiles = [len(image_grid_thw)]
        else:
            raise NotImplementedError(f"Unknown training phase {self.args.training_phase}")

        if len(input_ids) == 0:
            raise ValueError(f"input_ids is empty in {sample.__key__}")

        if self.args.enable_discard_sample:
            assert len(input_ids) <= self.args.seq_length, f"{sample.__key__} input length {len(input_ids)}"
        elif sample.video is not None:
            assert video_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length, (
                f"{sample.__key__} grid_thw: {video_grid_thw}"
            )
        elif sample.image is not None:
            image_token_len = int(image_grid_thw.prod(dim=-1).sum().item() / 4)
            assert image_token_len <= self.args.seq_length, (
                "Image token length exceeds seq_length:\n"
                f"- sample: {sample.__key__}\n"
                f"- image_grid_thw: {image_grid_thw}\n"
                f"- image_token_len: {image_token_len}\n"
                f"- seq_length: {self.args.seq_length}"
            )

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
            patch_positions=patch_positions,
        )

    def process_samples_grid(self, samples):
        """concat grid_thw for image and video"""
        image_grid_thw = [x.image_grid_thw for x in samples if x.image_grid_thw is not None]
        video_grid_thw = [x.video_grid_thw for x in samples if x.video_grid_thw is not None]

        if len(image_grid_thw) > 0:
            image_grid_thw = torch.cat(image_grid_thw).to(dtype=torch.int32)
        else:
            image_grid_thw = None

        if len(video_grid_thw) > 0:
            video_grid_thw = torch.cat(video_grid_thw).to(dtype=torch.int32)
        else:
            video_grid_thw = None

        return image_grid_thw, video_grid_thw
    
    @override
    @stateless
    def pack_selected_samples(self, samples: list[Qwen2VLImageTaskSample]) -> list[Qwen2VLImageTaskSamplePacked]:
        """Pack selected samples into one big sample."""
        image_grid_thw, video_grid_thw = self.process_samples_grid(samples)
        return Qwen2VLImageTaskSamplePacked(
            super().pack_selected_samples(samples), image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw
        )

    @override
    def batch(
        self, samples: list[Union[Qwen2VLImageTaskSample, Qwen2VLImageTaskSamplePacked]]
    ) -> Qwen2VLImageTaskBatchPacked:
        """Batch samples together"""
        image_grid_thw, video_grid_thw = self.process_samples_grid(samples)
        return Qwen2VLImageTaskBatchPacked(
            super().batch(samples), image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw
        )

    @override
    def process_images(
        self, samples: list[Union[Qwen2VLImageTaskSample, Qwen2VLImageTaskSamplePacked]]
    ) -> torch.Tensor:
        """ " Process the data to get the model's input"""
        imgs = [img for s in samples if s.imgs is not None for img in s.imgs]
        if len(imgs) > 0:
            return torch.cat(imgs)
        else:
            return torch.tensor([[0]], dtype=torch.float32)

    @override
    def process_videos(
        self, samples: list[Union[Qwen2VLImageTaskSample, Qwen2VLImageTaskSamplePacked]]
    ) -> torch.Tensor:
        """ " Process the data to get the model's input"""
        pixel_values_videos = [
            pixel_values_video
            for s in samples
            if s.pixel_values_videos is not None
            for pixel_values_video in s.pixel_values_videos
        ]
        if len(pixel_values_videos) > 0:
            return torch.cat(pixel_values_videos)
        else:
            return torch.tensor([[0]], dtype=torch.float32)

    @override
    def build_train_datasets(
        self,
        *,
        datasets: list[tuple[BaseCoreDatasetFactory[T_sample], Union[float, int, None]]],
        worker_config: WorkerConfig,
        batch_size: Optional[int],
        batch_drop_last: bool = False,
        packing_buffer_size: Optional[int] = None,
        virtual_epoch_length: int = 0,
        shuffle_buffer_size: Optional[int] = None,
        blend_mode: DatasetBlendMode = DatasetBlendMode.NONE,
        repeat: bool = True,
    ) -> SavableDataset[T_batch]:
        """Combines train datasets to a single dataset."""

        # Check if there's a CrudeWebdataset but no cookers
        for dataset, _ in datasets:
            if isinstance(dataset, CrudeWebdataset):
                assert self.cookers, "CrudeWebdataset found, but no cookers registered."

        global_workers = max(1, worker_config.num_workers) * worker_config.world_size
        rotation_lengths = [len(dataset) for dataset, _ in datasets]
        for i in range(1, len(rotation_lengths)):
            rotation_lengths[i] += rotation_lengths[i - 1]
        worker_rotation_offsets = [rotation_length % global_workers for rotation_length in [0] + rotation_lengths[:-1]]

        if repeat:
            inner_datasets = [
                (
                    RepeatDataset(
                        dataset.build(worker_rotation_offset=worker_rotation_offset),
                        worker_config=worker_config,
                    ),
                    1.0 if weight is None else float(weight),
                )
                for (dataset, weight), worker_rotation_offset in zip(datasets, worker_rotation_offsets)
            ]
        else:
            assert blend_mode in (
                DatasetBlendMode.NONE,
                DatasetBlendMode.SAMPLE_REPETITIONS,
            ) and all(isinstance(repetitions, int) for _dataset, repetitions in datasets), (
                "If repeat is False, the datasets must be repeated with integer weights."
            )
            inner_datasets = [
                (
                    (
                        dataset.build(worker_rotation_offset=worker_rotation_offset)
                        if repetition is None or repetition == 1
                        else RepeatDataset(
                            dataset.build(worker_rotation_offset=worker_rotation_offset),
                            repeats=int(repetition),
                            worker_config=worker_config,
                        )
                    ),
                    len(dataset) * (1 if repetition is None else int(repetition)),
                )
                for (dataset, repetition), worker_rotation_offset in zip(datasets, worker_rotation_offsets)
            ]

        if len(inner_datasets) > 1:
            # The worker offset for each dataset is the cumsum of the dataset lengths, but modulo the
            # global number of workers.
            dataset = BlendDataset(
                *inner_datasets,
                worker_config=worker_config,
            )
        elif len(datasets) == 1:
            dataset = inner_datasets[0][0]
        else:
            raise ValueError("No datasets given.")
        if shuffle_buffer_size is not None and shuffle_buffer_size > 1:
            dataset = ShuffleBufferDataset(
                dataset,
                size=shuffle_buffer_size,
                worker_config=worker_config,
            )
        dataset = self.build_cook_crude_sample(dataset, worker_config=worker_config)
        dataset = self.build_encode_sample(dataset, worker_config=worker_config)

        # Insert pool sorting before entering BatchDataset
        if getattr(self.args, "length_sort_pool_size", 0) and self.args.length_sort_pool_size > 0:
            dataset = LengthPoolSortDataset(
                dataset,
                pool_size=self.args.length_sort_pool_size,
                key_fn=lambda s: getattr(s, "total_len", len(getattr(s, "tokens"))),
                ascending=not getattr(self.args, "length_sort_desc", False),
                worker_config=worker_config,
            )
        dataset = self.build_batch(
            dataset,
            batch_size=batch_size,
            batch_drop_last=batch_drop_last,
            packing_buffer_size=packing_buffer_size,
            worker_config=worker_config,
        )
        if virtual_epoch_length > 0:
            dataset = EpochizeDataset(
                dataset,
                length=virtual_epoch_length,
                worker_config=worker_config,
            )
        if worker_config.should_log(level=1):
            dataset = LogSampleDataset(dataset, mode="train", worker_config=worker_config)
        return dataset
