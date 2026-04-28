from __future__ import annotations
import unicodedata, re
from typing import Dict, Any


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(text: str, cfg: Dict[str, Any]) -> str:
    norm_cfg = cfg.get("normalization", {})

    # Unicode normalization
    nf = norm_cfg.get("unicode_nf", None)
    if nf:
        text = unicodedata.normalize(nf, text)

    # casefold
    if norm_cfg.get("casefold", False):
        text = text.casefold()

    # ligatures
    if norm_cfg.get("normalize_ligatures", False):
        text = text.replace("æ", "ae").replace("œ", "oe")

    # u/v
    if norm_cfg.get("map_u_v", False):
        text = text.replace("v", "u")

    # i/j
    if norm_cfg.get("map_i_j", False):
        text = text.replace("j", "i")

    # diacritics
    if norm_cfg.get("strip_diacritics", False):
        text = strip_diacritics(text)

    text = re.sub(r'([()\[\]{}“”‘’\'"«»])', r' \1 ', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text