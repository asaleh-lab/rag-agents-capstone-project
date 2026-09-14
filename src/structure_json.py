"""M1L1: structure California restaurant blurbs into validated JSON via OpenAI."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "California-Culinary-Map.txt"
OUTPUT_PATH = ROOT / "data" / "structured_restaurant_data.json"

load_dotenv(ROOT / ".env")


class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


def load_restaurant_paragraphs(file_path: Path) -> list[str]:
    data = file_path.read_text(encoding="utf-8")
    restaurant_list = data.split("\n\n")
    restaurant_list = restaurant_list[1:]
    return [p.strip() for p in restaurant_list if p.strip()]


def llm_model(system_msg: str, prompt_txt: str) -> str:
    client = OpenAI()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_txt},
        ],
    )
    return response.choices[0].message.content or ""


def safe_llm_call(system_msg: str, prompt_txt: str, retries: int = 3) -> str:
    for _ in range(retries):
        try:
            return llm_model(system_msg, prompt_txt)
        except Exception:
            time.sleep(2)
    return "Failed after retries"


EXAMPLE_OUTPUT = """{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
}"""


def restaurant_data_structure_prompt_generation(
    restaurant_paragraph: str, example_paragraph: str
) -> tuple[str, str]:
    base_system_msg = """
You extract structured restaurant attributes from unstructured description text.
Return ONLY a single valid JSON object. No markdown fences. No commentary.
Use these keys: name, location, type, food_style, rating, price_range,
signatures, vibe, environment, shortcomings.
rating is a float when known, otherwise null.
price_range is an integer 1-4 when known, otherwise null.
signatures and shortcomings are JSON arrays of strings.
""".strip()

    base_user_prompt = f"""
Task:
Extract the restaurant attributes from the description below into the JSON schema
shown in the example. Match the example's keys exactly.

Restaurant description:
{restaurant_paragraph}

Example:
Input Restaurant Description: {example_paragraph}
Output:
{EXAMPLE_OUTPUT}
""".strip()
    return base_system_msg, base_user_prompt


def JSON_auto_repair_prompts(
    candidate_json_output: str, error_message: str
) -> tuple[str, str]:
    auto_repair_system_msg = """
You repair malformed JSON so it matches a restaurant schema.
Return ONLY the corrected JSON object. No markdown fences. No commentary.
Required keys: name, location, type, food_style, rating, price_range,
signatures, vibe, environment, shortcomings.
""".strip()
    auto_repair_prompt = f"""
The following text was supposed to be valid restaurant JSON but failed validation.

Validation error:
{error_message}

Candidate text:
{candidate_json_output}

Return the fixed JSON object only.
""".strip()
    return auto_repair_system_msg, auto_repair_prompt


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def structure_one_restaurant(
    restaurant_paragraph: str, example_paragraph: str, max_repairs: int = 5
) -> str:
    system_msg, user_prompt = restaurant_data_structure_prompt_generation(
        restaurant_paragraph, example_paragraph
    )
    candidate = strip_code_fences(safe_llm_call(system_msg, user_prompt))

    attempts = 0
    while attempts <= max_repairs:
        try:
            Restaurant.model_validate_json(candidate)
            return candidate
        except ValidationError as exc:
            if attempts == max_repairs:
                raise
            repair_system, repair_prompt = JSON_auto_repair_prompts(
                candidate, exc.json()
            )
            candidate = strip_code_fences(
                safe_llm_call(repair_system, repair_prompt)
            )
            attempts += 1
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structure California Culinary Map text into JSON (M1L1)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N restaurants (smoke test).",
    )
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Download California-Culinary-Map.txt into data/."
        )

    restaurant_list = load_restaurant_paragraphs(DATA_PATH)
    if args.limit is not None:
        restaurant_list = restaurant_list[: args.limit]

    print(f"Restaurants to structure: {len(restaurant_list)}")
    print(restaurant_list[0][:200] if restaurant_list else "(empty)")

    example_paragraph = (
        restaurant_list[1] if len(restaurant_list) > 1 else restaurant_list[0]
    )

    structured_restaurant_lists: list[str] = []
    for i, restaurant_paragraph in enumerate(restaurant_list):
        finalized = structure_one_restaurant(restaurant_paragraph, example_paragraph)
        structured_restaurant_lists.append(finalized)
        if (i + 1) % 20 == 0:
            print(f"{i + 1} out of {len(restaurant_list)} is done")

    print("ALL DONE!!")

    structured_restaurant_lists_json = [
        json.loads(response) for response in structured_restaurant_lists
    ]
    for i, response in enumerate(structured_restaurant_lists_json):
        response["itemId"] = 1000001 + i
        structured_restaurant_lists_json[i] = response

    if structured_restaurant_lists_json:
        index = min(49, len(structured_restaurant_lists_json) - 1)
        print(f"Sample item at index {index}:")
        print(json.dumps(structured_restaurant_lists_json[index], indent=2))

    OUTPUT_PATH.write_text(
        json.dumps(structured_restaurant_lists_json, indent=4),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
