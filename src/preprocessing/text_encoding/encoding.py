from collections import Counter
from typing import Iterable, List, Dict


def tokenize(text: str) -> List[str]:
    """
    Tokenize a string into a list of lowercase words.

    Args:
        text (str): Input text string.

    Returns:
        List[str]: List of lowercase tokens.
    """
    return text.lower().split()


def encode(text: str, vocab: Dict[str, int], max_len: int = 50) -> List[int]:
    """
    Encode a text string into a list of token IDs, padding or truncating to max_len.

    Args:
        text (str): Input text string.
        vocab (Dict[str, int]): Vocabulary mapping tokens to IDs.
        max_len (int, optional): Maximum sequence length. Defaults to 50.

    Returns:
        List[int]: List of token IDs of length max_len.
    """
    tokens = tokenize(text)
    ids = [vocab.get(t, 0) for t in tokens]

    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def create_vocab(documents: Iterable[str]) -> Dict[str, int]:
    """
    Create a vocabulary from an iterable of documents, mapping most common words to integer IDs.

    Args:
        documents (Iterable[str]): Iterable of text documents.

    Returns:
        Dict[str, int]: Vocabulary mapping tokens to unique integer IDs (starting from 1).
                        ID 0 is reserved for unknown tokens/padding.
    """
    counter = Counter()
    for doc in documents:
        counter.update(tokenize(doc))

    vocab = {word: idx+1 for idx, (word, _) in enumerate(counter.most_common())}
    return vocab
