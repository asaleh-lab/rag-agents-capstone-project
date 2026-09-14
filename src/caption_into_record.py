"""M1L2: caption recipe and review images, then merge into structured JSON."""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECIPES_PATH = DATA / "Recipes.json"
REVIEWS_PATH = DATA / "Synthetic-User-Reviews.json"
IMAGES_DIR = DATA / "synthetic_recipe_images"
ZIP_PATH = DATA / "synthetic-recipe-images.zip"
PLACEHOLDER = DATA / "review_image_placeholder.jpg"
OUT_RECIPES = DATA / "augmented_food_recipe.json"
OUT_REVIEWS = DATA / "augmented_user_review.json"

RECIPES_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hpTjb6liKBLVHQK0UgMi5A/Recipes.json"
REVIEWS_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fQUs9wQ6aB6ts6fmkD2V2w/Synthetic-User-Reviews.json"
IMAGES_ZIP_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"

load_dotenv(ROOT / ".env")


def ensure_data() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if not RECIPES_PATH.exists():
        urllib.request.urlretrieve(RECIPES_URL, RECIPES_PATH)
    if not REVIEWS_PATH.exists():
        urllib.request.urlretrieve(REVIEWS_URL, REVIEWS_PATH)
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.glob("*.png")):
        if not ZIP_PATH.exists():
            print("Downloading recipe images zip (large)...")
            urllib.request.urlretrieve(IMAGES_ZIP_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA)


def vision_llm(system_msg: str, prompt_txt: str, image_path: Path) -> str:
    client = OpenAI()
    model = os.environ.get("OPENAI_VISION_MODEL") or os.environ.get(
        "OPENAI_MODEL", "gpt-4o-mini"
    )
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_txt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            },
        ],
    )
    return response.choices[0].message.content or ""


def image_caption_prompt_template(food_name: str) -> tuple[str, str]:
    image_caption_system_msg = """
You caption food photos for a restaurant recommendation knowledge base.
Describe ingredients, presentation, and cooking style in 2-4 clear sentences.
Focus on the named dish. No markdown. No preamble.
""".strip()
    image_caption_prompt_txt = f"""
Write a concise caption for this image of: {food_name}.
Mention visible ingredients, plating, and style that would help retrieval later.
""".strip()
    return image_caption_system_msg, image_caption_prompt_txt


def review_context_image_caption_prompt_template(reviews: str) -> tuple[str, str]:
    review_context_image_caption_system_msg = """
You caption dining photos using the diner's review as context.
Describe what is visible and how it relates to the review.
2-4 clear sentences. No markdown. No preamble.
""".strip()
    review_context_image_caption_prompt_txt = f"""
User review:
{reviews}

Caption the attached dining photo in light of that review.
""".strip()
    return (
        review_context_image_caption_system_msg,
        review_context_image_caption_prompt_txt,
    )


def recipe_image_path(recipe_id: int) -> Path:
    return IMAGES_DIR / f"recipe{recipe_id}.png"


def caption_recipes(recipe_data: list[dict], limit: int | None) -> list[dict]:
    items = recipe_data if limit is None else recipe_data[:limit]
    for i, recipe in enumerate(items):
        if (i + 1) % 20 == 0:
            print(f"{i + 1} out of {len(items)} is done")
        food_name = recipe["name"]
        image_path = recipe_image_path(recipe["id"])
        if not image_path.exists():
            recipe["image_description"] = f"(missing image for {image_path.name})"
            continue
        system_msg, prompt_txt = image_caption_prompt_template(food_name)
        recipe["image_description"] = vision_llm(system_msg, prompt_txt, image_path)
    print("ALL DONE!")
    return items if limit is None else recipe_data


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_data_with_retry(url: str) -> requests.Response:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response


def caption_reviews(user_review_data: list[dict], limit: int | None) -> list[dict]:
    items = user_review_data if limit is None else user_review_data[:limit]
    for i, review in enumerate(items):
        review_images = ast.literal_eval(review["images"])
        review_image_captions: list[str] = []
        if review_images:
            system_msg, prompt_txt = review_context_image_caption_prompt_template(
                review["text"]
            )
            for img_url in review_images:
                try:
                    image_data = get_data_with_retry(img_url)
                    print("Success!")
                except Exception as exc:
                    print(f"All retries failed at url {img_url}:", exc)
                    continue
                PLACEHOLDER.write_bytes(image_data.content)
                caption = vision_llm(system_msg, prompt_txt, PLACEHOLDER)
                review_image_captions.append(caption)
        review["image_captions"] = review_image_captions
    print("ALL DONE!")
    return user_review_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caption recipe and review images into JSON (M1L2)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N recipes and reviews (smoke test).",
    )
    parser.add_argument(
        "--skip-reviews",
        action="store_true",
        help="Only caption recipes (for M1L2_caption_all_recipes screenshot).",
    )
    args = parser.parse_args()

    ensure_data()
    recipe_data = json.loads(RECIPES_PATH.read_text(encoding="utf-8"))
    print("First recipe:")
    first = recipe_data[0]
    for key, value in first.items():
        print(f"{key} ({type(value).__name__}): {value}")

    print("\nCaptioning recipes...")
    caption_recipes(recipe_data, args.limit)
    OUT_RECIPES.write_text(json.dumps(recipe_data, indent=4), encoding="utf-8")
    print(f"Wrote {OUT_RECIPES}")

    if args.skip_reviews:
        return

    user_review_data = json.loads(REVIEWS_PATH.read_text(encoding="utf-8"))
    print("\nFirst review:")
    for key, value in user_review_data[0].items():
        print(f"{key} ({type(value).__name__}): {value}")

    print("\nCaptioning review images...")
    caption_reviews(user_review_data, args.limit)
    OUT_REVIEWS.write_text(json.dumps(user_review_data, indent=4), encoding="utf-8")
    print(f"Wrote {OUT_REVIEWS}")


if __name__ == "__main__":
    main()
