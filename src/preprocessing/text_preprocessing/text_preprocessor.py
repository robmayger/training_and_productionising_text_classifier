import re
from typing import List
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from textblob import TextBlob

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

class TextPreprocessor:
    """
    Preprocesses text data for NLP tasks.

    Features:
    - Lowercasing
    - Noise removal (URLs, emails, phone numbers, punctuation)
    - Tokenization
    - Stopword removal
    - Lemmatization
    - Number normalization
    - Optional spelling correction
    - Optional bigram detection and merging
    """
    def __init__(
        self, use_spell_correction: bool = False, 
        min_bigram_count: int = 5, bigram_threshold: float = 10.0
    ) -> None:
        self.use_spell_correction = use_spell_correction
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])
        self.bigrams = None
        self.min_bigram_count = min_bigram_count
        self.bigram_threshold = bigram_threshold

    def clean_text(self, text: str) -> str:
        """
        Clean text by lowercasing and removing noise.

        Removes:
        - URLs
        - Emails
        - Phone numbers
        - Non-alphanumeric characters
        - Extra whitespace

        Args:
            text (str): Input text.

        Returns:
            str: Cleaned text.
        """
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\+?\d[\d -]{8,}\d', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove common stopwords"""
        return [word for word in tokens if word not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens using spaCy"""
        doc = self.nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]

    def normalize_numbers(self, tokens: List[str]) -> List[str]:
        """Replace digits with <NUM> token"""
        return ['<NUM>' if token.isdigit() else token for token in tokens]

    def correct_spelling(self, text: str) -> str:
        """Optional spelling correction using TextBlob"""
        return str(TextBlob(text).correct())

    def build_bigrams(self, token_lists: List[List[str]]):
        """
        Train a bigram model on a list of tokenized documents.

        Args:
            token_lists (List[List[str]]): List of tokenized documents.
        """
        all_tokens = [token for tokens in token_lists for token in tokens]
        finder = BigramCollocationFinder.from_words(all_tokens)
        finder.apply_freq_filter(self.min_bigram_count)
        scored = finder.score_ngrams(BigramAssocMeasures.pmi)

        # Store top bigrams above threshold
        self.bigrams = {
            f"{w1}_{w2}" for (w1, w2), score in scored if score > self.bigram_threshold
        }

    def apply_bigrams(self, tokens: List[str]) -> List[str]:
        """
        Merge known bigrams in a list of tokens.

        Args:
            tokens (List[str]): Tokenized text.

        Returns:
            List[str]: Tokens with bigrams merged.
        """
        if not self.bigrams:
            return tokens

        merged_tokens = []
        skip_next = False
        for i in range(len(tokens) - 1):
            if skip_next:
                skip_next = False
                continue
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            if bigram in self.bigrams:
                merged_tokens.append(bigram)
                skip_next = True
            else:
                merged_tokens.append(tokens[i])
        if not skip_next:
            merged_tokens.append(tokens[-1])
        return merged_tokens

    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to a text string.

        Steps:
        1. Optional spelling correction
        2. Cleaning (lowercase, remove noise)
        3. Tokenization
        4. Stopword removal
        5. Lemmatization
        6. Number normalization
        7. Bigram merging

        Args:
            text (str): Input text.

        Returns:
            str: Preprocessed text as a single string.
        """
        if self.use_spell_correction:
            text = self.correct_spelling(text)
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        tokens = self.normalize_numbers(tokens)
        tokens = self.apply_bigrams(tokens)
        return " ".join(tokens)
