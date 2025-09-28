from collections import Counter
from typing import Iterable


def tokenize(text):
    return text.lower().split()


def encode(text: list[str], vocab: dict[str, int], max_len=50):
    tokens = tokenize(text)
    ids = [vocab.get(t, 0) for t in tokens]

    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def create_vocab(documents: Iterable[list[str]]):
    counter = Counter()
    for doc in documents:
        counter.update(tokenize(doc))

    vocab = {word: idx+1 for idx, (word, _) in enumerate(counter.most_common())}
    return vocab
