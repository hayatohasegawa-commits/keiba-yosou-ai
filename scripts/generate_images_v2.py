"""Dark dramatic hero images: running horses.

参考: Rebike-AI のシネマティックヒーロー + ダークモードで再構築。
"""
from __future__ import annotations

import base64
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from openai import OpenAI

OUT_DIR = ROOT / "app" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGES = [
    {
        "name": "hero_dark.png",
        "size": "1536x1024",
        "quality": "high",
        "prompt": (
            "Cinematic ultra-wide hero image of a magnificent thoroughbred racehorse "
            "galloping at full speed, dramatic side profile in motion, powerful muscles, "
            "flowing mane, dynamic motion blur on background, "
            "deep black background (#0a0a0f), warm golden rim light on the horse, "
            "subtle emerald data lines flowing horizontally in the background, "
            "minimal premium editorial style, photorealistic, "
            "dramatic cinematic lighting, sophisticated, professional sports brand "
            "aesthetic, no text, no logos, no rider, single horse only, "
            "full bleed composition with horse occupying center-right, "
            "negative space on left for typography"
        ),
    },
    {
        "name": "hero_pack.png",
        "size": "1536x1024",
        "quality": "high",
        "prompt": (
            "Cinematic shot of multiple racehorses galloping together on a dark "
            "racetrack, dramatic motion blur, low angle dynamic, "
            "deep black background fading to navy (#0a0a0f to #0a2540), "
            "warm gold dust kicked up from hooves, emerald track lights in background, "
            "Bloomberg-meets-sports-brand premium editorial style, photorealistic, "
            "dramatic cinematic, no text, no logos, no riders, "
            "ultra-wide composition, sophisticated mood"
        ),
    },
    {
        "name": "section_horse.png",
        "size": "1024x1024",
        "quality": "medium",
        "prompt": (
            "Minimal premium portrait of a noble racehorse head in profile, "
            "deep black background (#0a0a0f), subtle golden rim lighting on the horse, "
            "elegant sophisticated mood, editorial premium magazine style, "
            "photorealistic, no text, no bridle, single horse head centered, "
            "negative space, premium sports brand aesthetic"
        ),
    },
]


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    client = OpenAI()
    for img in IMAGES:
        out_path = OUT_DIR / img["name"]
        if out_path.exists():
            print(f"[skip] {img['name']}")
            continue
        print(f"[generate] {img['name']} ({img['size']}, {img['quality']})...")
        try:
            resp = client.images.generate(
                model="gpt-image-1",
                prompt=img["prompt"],
                size=img["size"],
                quality=img["quality"],
                n=1,
            )
            data0 = resp.data[0]
            if getattr(data0, "b64_json", None):
                with out_path.open("wb") as f:
                    f.write(base64.b64decode(data0.b64_json))
            else:
                with urllib.request.urlopen(data0.url) as r, out_path.open("wb") as f:
                    f.write(r.read())
            print(f"  saved: {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
