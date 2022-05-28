import spacy
import tqdm

from .tokenizer import Tokenizer


class SimpleTokenizer(Tokenizer):
    """Simple tokenizer with lowercasing, stop-word and punctuation removal,
    and optional lemmatization using spacy.
    """

    def __init__(self, lemmatization=False, verbose=1):
        """Initialize SimpleTokenizer.

        Args:
            lemmatization (bool, optional): True if lemmatization should be
             performed, false otherwise. Defaults to False.
        """
        self.lemmatization = lemmatization
        self.verbose = verbose
        exclude = ["parser", "ner"]
        # Exclude tagger when lemmatization is not performed.
        if not self.lemmatization:
            exclude.append("tagger")
        self.nlp = spacy.load("en_core_web_sm", exclude=exclude)

    def _tokenize_doc(self, doc):
        tokens = []
        for text in doc:
            token = text.lemma_ if self.lemmatization else text.text
            if not (text.is_stop or text.is_punct or text.is_space):
                tokens.append(token.strip().lower())
        return tokens

    def name(self):
        return self.__class__.__name__ + "_" + (
            "lemmatize" if self.lemmatization else "no_lemmatize")

    def tokenize(self, sentence):
        return self._tokenize_doc(self.nlp(sentence))

    def tokenize_batch(self, sentences):
        tokenized_sentences = []
        for doc in tqdm.tqdm(self.nlp.pipe(sentences,
                                           batch_size=4000,
                                           n_process=4),
                             desc="Tokenizing data...",
                             disable=self.verbose == 0,
                             total=len(sentences)):
            tokenized_sentences.append(self._tokenize_doc(doc))
        return tokenized_sentences
