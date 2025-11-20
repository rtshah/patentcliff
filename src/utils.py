"""Utility functions for config loading, date parsing, and common operations."""
import yaml
from pathlib import Path
from datetime import date, datetime
from typing import Optional
import re


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_date_flexible(date_str: Optional[str]) -> Optional[date]:
    """
    Parse date string with multiple format attempts.
    Handles: "Aug 24, 2026", "August 24, 2026", "08/24/2026", "2026-08-24"
    Returns None for empty/NA/N/A/null values.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    date_str = date_str.strip()
    if not date_str or date_str.upper() in ("NA", "N/A", "NULL", ""):
        return None
    
    formats = [
        "%b %d, %Y",      # Aug 24, 2026
        "%B %d, %Y",      # August 24, 2026
        "%m/%d/%Y",       # 08/24/2026
        "%Y-%m-%d",       # 2026-08-24
        "%d/%m/%Y",       # 24/08/2026 (alternative)
        "%Y/%m/%d",       # 2026/08/24
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    # Last attempt: try to extract date-like patterns
    match = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', date_str)
    if match:
        try:
            year, month, day = map(int, match.groups())
            return date(year, month, day)
        except ValueError:
            pass
    
    return None


def normalize_column_name(col: str) -> str:
    """Normalize column name to lowercase with underscores."""
    return col.strip().lower().replace(" ", "_").replace("-", "_")


def find_column(df_columns: list, aliases: list[str]) -> Optional[str]:
    """Find column in dataframe by checking aliases (case-insensitive)."""
    normalized_cols = {normalize_column_name(c): c for c in df_columns}
    for alias in aliases:
        normalized_alias = normalize_column_name(alias)
        if normalized_alias in normalized_cols:
            return normalized_cols[normalized_alias]
    return None


def canonicalize_route(route_text: str) -> str:
    """Canonicalize route text to standard form."""
    if not route_text:
        return ""
    
    route = route_text.upper().strip()
    route = re.sub(r'\s+', ' ', route)
    
    # Common mappings
    mappings = {
        "PO": "ORAL",
        "IV": "INTRAVENOUS",
        "IM": "INTRAMUSCULAR",
        "SQ": "SUBCUTANEOUS",
        "SC": "SUBCUTANEOUS",
        "TOP": "TOPICAL",
    }
    
    for abbrev, canonical in mappings.items():
        if route == abbrev or route.startswith(abbrev + " "):
            return canonical
    
    return route


def canonicalize_dosage_form(form_text: str) -> str:
    """Canonicalize dosage form text to standard form."""
    if not form_text:
        return ""
    
    form = form_text.upper().strip()
    form = re.sub(r'\s+', ' ', form)
    
    # Remove common suffixes for matching
    form_base = form.split(",")[0].strip()  # "AEROSOL, FOAM" -> "AEROSOL"
    
    # Common mappings
    mappings = {
        "SOLN": "SOLUTION",
        "SUSP": "SUSPENSION",
        "TAB": "TABLET",
        "CAP": "CAPSULE",
        "INJ": "INJECTION",
    }
    
    for abbrev, canonical in mappings.items():
        if form_base == abbrev or form_base.startswith(abbrev + " "):
            return canonical
    
    return form_base


def normalize_strength_for_matching(strength: str) -> str:
    """
    Normalize strength string for SCD matching.
    Removes leading 'EQ ' and trailing descriptors, normalizes units and separators.
    """
    if not strength:
        return ""
    
    s = strength.upper().strip()
    s = re.sub(r'\s+', ' ', s)
    
    # Remove leading "EQ "
    s = re.sub(r'^EQ\s+', '', s)
    
    # Remove trailing descriptors (BASE, ACID, HCL, etc.)
    s = re.sub(r'\s+(BASE|ACID|HCL|HYDROCHLORIDE|SALT)$', '', s, flags=re.IGNORECASE)
    
    # Normalize separators
    s = re.sub(r'\s+PER\s+', '/', s)
    
    # Normalize units (ensure space before unit)
    s = re.sub(r'(\d+)(MCG|MG|G|ML|L|%|UNIT)', r'\1 \2', s)
    
    return s.strip()

