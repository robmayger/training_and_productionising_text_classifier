from .mean_pooling_text_classifier import MeanPoolingTextClassifier
from .transformer_text_classifier import TransformerTextClassifier

MODEL_REGISTRY = {
    "MeanPoolingTextClassifier": MeanPoolingTextClassifier,
    "TransformerTextClassifier": TransformerTextClassifier,
}