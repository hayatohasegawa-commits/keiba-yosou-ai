"""OpenAI 画像生成: ヒーロー画像、OG画像、背景パターン、勝利アイコン。

実行:
    python scripts/generate_images.py
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
        "name": "hero_banner.png",
        "size": "1536x1024",
        "quality": "high",
        "prompt": (
            "Ultra minimal premium banner illustration: elegant abstract horse silhouette "
            "galloping through subtle digital data grid lines, deep navy background "
            "(#0a2540), thin emerald (#10b981) and gold (#d4af37) accent lines, "
            "Bloomberg terminal aesthetic meets Japanese minimal design, ultra clean "
            "negative space, sophisticated, professional analytics dashboard hero, "
            "no text, no logos, cinematic widescreen composition"
        ),
    },
    {
        "name": "og_image.png",
        "size": "1536x1024",
        "quality": "high",
        "prompt": (
            "Premium horse racing analytics hero image, abstract horse profile in "
            "elegant gold line art, scattered emerald data points and small charts, "
            "deep navy background (#0a2540), minimal premium design, Apple-style "
            "negative space, no readable text, no logos, sophisticated, intelligent, "
            "trustworthy mood, social media preview aesthetic"
        ),
    },
    {
        "name": "background_pattern.png",
        "size": "1024x1024",
        "quality": "medium",
        "prompt": (
            "Extremely subtle minimal seamless pattern: Japanese seigaiha wave motif "
            "intertwined with elegant racetrack curves, very low contrast cream "
            "(#fafaf7) with barely-there navy lines (#0a2540 at 6 percent opacity), "
            "delicate, refined, premium texture, no text, perfect for background use"
        ),
    },
    {
        "name": "trophy_icon.png",
        "size": "1024x1024",
        "quality": "medium",
        "prompt": (
            "Minimal premium icon: elegant gold trophy with subtle emerald sparkle, "
            "deep navy background (#0a2540), modern flat design with very thin lines, "
            "centered composition, premium aesthetic for a successful prediction icon, "
            "no text, no extra elements, clean isolated icon"
        ),
    },
]


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    client = OpenAI()
    total_cost_est_usd = 0.0
    for img in IMAGES:
        out_path = OUT_DIR / img["name"]
        if out_path.exists():
            print(f"[skip] {img['name']} already exists")
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
                url = data0.url
                with urllib.request.urlopen(url) as r, out_path.open("wb") as f:
                    f.write(r.read())
            cost = 0.08 if img["quality"] == "hd" else 0.04
            total_cost_est_usd += cost
            print(f"  saved: {out_path}  (estimated ${cost})")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n推定合計コスト: ${total_cost_est_usd:.2f}")


if __name__ == "__main__":
    main()
