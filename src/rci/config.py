"""Paths and constants shared by every module. Nothing here runs anything."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SQL_DIR = ROOT / "sql"

# ── data source ──────────────────────────────────────────────────────────────
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_XLSX = RAW_DIR / "online_retail_II.xlsx"

SEED = 42

# ── cleaning rules (see pipeline.py for the reason behind each one) ───────────
PRICE_OUTLIER_RATIO = 20      # a price more than 20x the product's median price is a typo
MAX_LINE_QUANTITY = 50_000    # the only two lines above this were cancelled the same day

# ── repurchase model ─────────────────────────────────────────────────────────
HORIZON_DAYS = 90
# Each date is a "pretend today". Features use only data BEFORE it; the label is whether the
# customer buys in the 90 days AFTER it. Every split's label window ends before the next
# split's cutoff, so nothing from a later period leaks into an earlier one (tests enforce this).
SPLITS = {
    "train": ["2010-06-09", "2010-09-09", "2010-12-09", "2011-03-09"],
    "valid": ["2011-06-09"],
    "test": ["2011-09-09"],
}

# ── AI analyst ───────────────────────────────────────────────────────────────
GROQ_MODEL = "openai/gpt-oss-120b"
ANALYST_MAX_ROWS = 1000
ANALYST_TIMEOUT_S = 15


def load_env() -> None:
    """Load GROQ_API_KEY from `.env` in the project root, or from the file named by RCI_ENV_FILE."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # the key can still come from the real environment
        return
    load_dotenv(ROOT / ".env")
    extra = os.getenv("RCI_ENV_FILE")
    if extra:
        load_dotenv(extra)
