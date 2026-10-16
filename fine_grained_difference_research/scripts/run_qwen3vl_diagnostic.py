#!/usr/bin/env python3
"""Run the already-cached general VLM on a small synthetic pair diagnostic.

This is a zero-shot diagnostic, not an IDC specialist reproduction or benchmark.
The model is loaded only from the existing Hugging Face cache.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    "/home/vsu/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
    "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
SIZE = 448


def scene() -> Image.Image:
    image = Image.new("RGB", (SIZE, SIZE), (235, 233, 224))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 350, SIZE, SIZE), fill=(192, 180, 158))
    draw.line((0, 350, SIZE, 350), fill=(95, 88, 78), width=3)
    return image


def text_font(size: int = 78) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def build_cases(out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []

    unchanged = scene()
    d = ImageDraw.Draw(unchanged)
    d.rectangle((92, 188, 240, 286), fill=(42, 102, 174), outline=(30, 50, 70), width=4)
    d.ellipse((300, 225, 360, 285), fill=(225, 175, 45), outline=(100, 80, 25), width=3)
    cases.append(("no_change", unchanged.copy(), unchanged.copy(), "No depicted object or property changes."))

    before = scene()
    d = ImageDraw.Draw(before)
    d.rectangle((95, 192, 245, 286), fill=(42, 102, 174), outline=(30, 50, 70), width=4)
    after = before.copy()
    ImageDraw.Draw(after).ellipse((303, 223, 351, 271), fill=(222, 178, 40), outline=(95, 74, 20), width=3)
    cases.append(("small_addition", before, after, "A small yellow ball is added on the right."))
    cases.append(("small_removal_reversed", after.copy(), before.copy(),
                  "The small yellow ball is removed from the right."))

    before = scene()
    ImageDraw.Draw(before).rounded_rectangle((105, 205, 345, 320), radius=28,
                                             fill=(42, 102, 174), outline=(30, 50, 70), width=4)
    after = scene()
    ImageDraw.Draw(after).rounded_rectangle((105, 205, 345, 320), radius=28,
                                            fill=(215, 102, 45), outline=(30, 50, 70), width=4)
    cases.append(("color_change", before, after, "The large car-like object changes from blue to orange."))

    before = scene()
    ImageDraw.Draw(before).rectangle((90, 225, 160, 295), fill=(55, 150, 90), outline=(20, 70, 35), width=3)
    after = scene()
    ImageDraw.Draw(after).rectangle((270, 225, 340, 295), fill=(55, 150, 90), outline=(20, 70, 35), width=3)
    cases.append(("movement", before, after, "The green square moves from the left to the right."))
    cases.append(("movement_reversed", after.copy(), before.copy(),
                  "The green square moves from the right to the left."))

    before = scene()
    ImageDraw.Draw(before).rectangle((112, 190, 336, 315), fill=(245, 245, 242), outline=(65, 65, 65), width=5)
    ImageDraw.Draw(before).text((138, 205), "EXIT", font=text_font(76), fill=(25, 35, 45))
    after = before.copy()
    ImageDraw.Draw(after).rectangle((112, 190, 336, 315), fill=(245, 245, 242), outline=(65, 65, 65), width=5)
    ImageDraw.Draw(after).text((138, 205), "EX1T", font=text_font(76), fill=(25, 35, 45))
    cases.append(("text_edit", before, after, "The sign text changes from EXIT to EX1T."))

    before = scene()
    ImageDraw.Draw(before).rectangle((115, 210, 310, 320), fill=(70, 115, 160), outline=(35, 55, 80), width=4)
    after = ImageEnhance.Brightness(before).enhance(1.22)
    cases.append(("illumination_only", before, after, "No semantic change; the whole scene is brighter."))

    records = []
    for case_id, first, second, truth in cases:
        before_path = out_dir / f"{case_id}_before.png"
        after_path = out_dir / f"{case_id}_after.png"
        first.save(before_path)
        second.save(after_path)
        records.append({"case_id": case_id, "before": before_path, "after": after_path, "expected": truth})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path, default=ROOT / "results/predictions/qwen3vl_zero_shot_synthetic.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process; use the host execution environment.")
    if not args.model.is_dir():
        raise FileNotFoundError(f"Cached model directory not found: {args.model}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    case_list = build_cases(ROOT / "results/visualizations/qwen3vl_synthetic_pairs")
    records = []
    torch.cuda.reset_peak_memory_stats()
    for case in case_list:
        with Image.open(case["before"]) as im:
            before = im.convert("RGB")
        with Image.open(case["after"]) as im:
            after = im.convert("RGB")
        prompt = (
            "Compare the two images in order: image one is BEFORE and image two is AFTER. "
            "Describe only the meaningful visual change, including before-to-after direction. "
            "Ignore lighting-only variation. If nothing meaningful changed, say so. "
            "Use one concise sentence and do not guess details that are not visible."
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": before},
            {"type": "image", "image": after},
            {"type": "text", "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {key: value.to("cuda:0") if hasattr(value, "to") else value for key, value in inputs.items()}
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        answer = processor.batch_decode(trimmed, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)[0].strip()
        records.append({**case, "before": str(case["before"]), "after": str(case["after"]),
                        "prediction": answer, "latency_seconds": latency})
        print(f"{case['case_id']}: {answer} ({latency:.2f}s)", flush=True)

    result = {
        "model": "Qwen/Qwen3-VL-4B-Instruct (cached local checkpoint)",
        "checkpoint_path": str(args.model),
        "license": "Apache-2.0 per official model card; base model terms remain applicable",
        "mode": "zero-shot general VLM, not an image-difference specialist",
        "prompt_policy": prompt,
        "gpu": torch.cuda.get_device_name(0),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "cases": records,
        "scope": "small constructed qualitative diagnostic; not a benchmark or statistical evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {args.output}; peak VRAM {result['peak_cuda_memory_bytes'] / (1024**3):.2f} GiB")


if __name__ == "__main__":
    main()
