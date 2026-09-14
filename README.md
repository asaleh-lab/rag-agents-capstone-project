# rag-agents-capstone-project

IBM RAG and Agentic AI capstone rebuilt locally (Connoisseur Companion). OpenAI instead of watsonx. Same lab skills and course data.

## What we push

`src/`, input `data/` files that are small enough to version, `README.md`, `.env.example`, `requirements.txt`.

Not pushed: notebooks, assessment screenshots, generated JSON, recipe image zip/folders.

## How every module / lab works

1. Author drops the Coursera reading and/or downloaded `.ipynb` path in chat.
2. We add one `src/` script for that lab (OpenAI).
3. Script downloads any large assets on first run into `data/` (gitignored if huge).
4. Author runs the script locally, saves the named screenshot under `screenshots/` (gitignored).
5. Generated outputs stay local for later labs.
6. We commit the script + docs (+ small input data), not screenshots or notebooks.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put your OpenAI API key in `.env`.

## M1L1 — structure restaurant blurbs

```powershell
python src/structure_json.py
```

Screenshot: for-loop + `ALL DONE!!` → `screenshots/M1L1_structure_for_loop.jpg` (or `.png`).

## M1L2 — caption recipe and review images

Assessment screenshot can come from the **notebook** cell (preferred if you want the lab-shaped frame) or from the script. Notebook is local only (`notebooks/`, gitignored).

```powershell
jupyter lab notebooks/M1L2_Process_Multimodal_Data_with_LLMs.ipynb
```

Run top to bottom. When the recipe caption loop finishes (`ALL DONE!`), screenshot **that cell’s code + output** → `screenshots/M1L2_caption_all_recipes.jpg`.

Script equivalent (optional):

```powershell
python src/caption_into_record.py --limit 2
python src/caption_into_record.py --skip-reviews
python src/caption_into_record.py
```

Writes `data/augmented_food_recipe.json` and `data/augmented_user_review.json` (gitignored).
