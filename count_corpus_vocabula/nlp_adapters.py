from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AdapterWord:
    text: str
    lemma: Optional[str]
    upos: str

@dataclass
class AdapterSentence:
    words: List[AdapterWord]

@dataclass
class AdapterDoc:
    sentences: List[AdapterSentence]

class TransformersLatinAdapter:
    def __init__(self, model_name: str):
        from transformers import pipeline
        self.pos_pipeline = pipeline("token-classification", model=model_name)

    def __call__(self, text: str) -> AdapterDoc:
        hf_outputs = self.pos_pipeline(text)

        words = []
        for token in hf_outputs:
            word = AdapterWord(
                text=token["word"],
                lemma=token["word"].lower(),
                upos=self._map_to_upos(token["entity"])
            )
            words.append(word)

        sentence = AdapterSentence(words=words)
        return AdapterDoc(sentences=[sentence])

    def _map_to_upos(self, entity_tag: str) -> str:
        if "NOUN" in entity_tag: return "NOUN"
        if "VERB" in entity_tag: return "VERB"
        return "X"