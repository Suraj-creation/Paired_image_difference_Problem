"""Optional local-only Qwen3-VL adapter for ordered-pair descriptions."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image


PROMPT = """Compare the two images in order: first is BEFORE, second is AFTER.
Return only one JSON object with this schema:
{"status":"changed|no_meaningful_change|uncertain","changes":[{"type":"addition|removal|text_edit|color_material|shape_size|movement|orientation_pose|state|layout_structure|other","entity":"short visible entity name","before_state":"visible state or null","after_state":"visible state or null","changed_property":"specific change or null","spatial_description":"brief location in image or null","evidence":"only visible evidence","confidence":0.0}],"uncertainty":"brief caveat or null"}
Include multiple changes as separate entries. Only report differences directly visible in the pair; do not describe unrelated objects as changed. For movement, state only the observed source and destination. Do not infer from likely context. Treat lighting, shadows, reflections, camera movement, compression, or resizing as nuisance unless the depicted content itself changed. If text cannot be read precisely or evidence is ambiguous, return uncertain. Confidence is a subjective score, not calibrated. Do not include Markdown fences."""


class QwenPairCaptioner:
    """Load an already-cached checkpoint and describe one ordered image pair."""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", device: str = "cuda:0",
                 max_new_tokens: int = 384):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("CUDA is unavailable; run through the host GPU execution path or select CPU.")
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, local_files_only=True, dtype=dtype,
            device_map=device if device.startswith("cuda") else None,
            attn_implementation="sdpa",
        ).eval()
        # The shipped generation config has sampling-only values; neutralize
        # them for greedy decoding to keep the run deterministic and quiet.
        self.model.generation_config.temperature = 1.0
        self.model.generation_config.top_p = 1.0
        self.model.generation_config.top_k = 50
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        if not device.startswith("cuda"):
            self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)

    def resource_summary(self) -> dict[str, Any]:
        summary = {"device": self.device, "gpu_name": None, "peak_allocated_vram_bytes": None,
                   "checkpoint_revision": getattr(self.model.config, "_commit_hash", None)}
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            summary["gpu_name"] = self.torch.cuda.get_device_name(0)
            summary["peak_allocated_vram_bytes"] = int(self.torch.cuda.max_memory_allocated())
        return summary

    def describe(self, before_path: str | Path, after_path: str | Path) -> dict[str, Any]:
        with Image.open(before_path) as image:
            before = image.convert("RGB")
        with Image.open(after_path) as image:
            after = image.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": before}, {"type": "image", "image": after},
            {"type": "text", "text": PROMPT},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        inputs.pop("token_type_ids", None)
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        answer_ids = [out[len(inp):] for inp, out in zip(inputs["input_ids"], output)]
        raw = self.processor.batch_decode(answer_ids, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0].strip()
        parsed = _parse_json_object(raw)
        return {
            "model": self.model_id,
            "raw_output": raw,
            "parsed": parsed,
            "parse_valid": parsed is not None,
            "latency_seconds": elapsed,
            "confidence_is_calibrated": False,
        }


def _parse_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or not isinstance(value.get("changes"), list):
            continue
        original_status = value.get("status")
        if original_status in {"uncertain", "ambiguous"}:
            value["status"] = "uncertain"
        elif original_status in {"no_meaningful_change", "no_change", "unchanged"} and not value["changes"]:
            value["status"] = "no_meaningful_change"
        elif value["changes"]:
            value["status"] = "changed"
        else:
            continue
        if original_status != value["status"]:
            value["model_status_normalized_from"] = original_status
        value["changes"] = [item for item in value["changes"] if isinstance(item, dict)]
        if value["status"] == "no_meaningful_change":
            value["changes"] = []
        return value
    return None

