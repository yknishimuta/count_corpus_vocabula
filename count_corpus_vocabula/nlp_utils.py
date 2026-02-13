from __future__ import annotations
from typing import Optional, Dict
from nlpo_toolkit.nlp import build_stanza_pipeline, PackageType

def make_package(stanza_package: Optional[str]) -> Optional[Dict[str, str]]:
    if stanza_package is None:
        return None
    sp = stanza_package.lower()
    if sp == "perseus":
        return {"tokenize": "perseus", "mwt": "perseus", "pos": "perseus", "lemma": "perseus"}
    return None

def build_pipeline(
    language: str = "la",
    stanza_package: PackageType = "perseus",
    cpu_only: bool = True,
):
    processors = "tokenize,mwt,pos,lemma"
    nlp = build_stanza_pipeline(
        lang=language,
        processors=processors,
        package=stanza_package,
        use_gpu=not cpu_only,
    )
    return nlp, stanza_package
