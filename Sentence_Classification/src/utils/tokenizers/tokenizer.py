from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract class for tokenizers."""

    @abstractmethod
    def name(self):
        """Name of the tokenizer."""

    @abstractmethod
    def tokenize(self, sentence):
        """Tokenize a single sentence.

        Args:
            sentence (str): Sentence to tokenize.
        """

    @abstractmethod
    def tokenize_batch(self, sentences):
        """Batch tokenize a list of sentences.

        Args:
            sentences (list[str]): List of sentences to tokenize.
        """
