from nlpo_toolkit.models import NLPDocument, NLPSentence, NLPToken

class FakeNLP:
    """
    空白区切りでトークン化し、すべてNOUNとして共通データモデルを返すダミー。
    """
    def __call__(self, text: str) -> NLPDocument:
        s = text
        tokens = []

        i = 0
        for raw in s.split():
            start = s.find(raw, i)
            if start < 0:
                start = i
            i = start + len(raw)

            tok = NLPToken(text=raw, lemma=raw.lower(), upos="NOUN", start_char=start)
            tokens.append(tok)

        sent = NLPSentence(text=s, tokens=tokens)
        return NLPDocument(sentences=[sent])