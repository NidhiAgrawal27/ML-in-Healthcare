import pickle
import os
import tqdm
import numpy as np


class PubMedDataset:
    """PubMed RCT dataset.

    Attributes:
        data_path (str): Path to data.
        verbose (int): Verbosity level.
        tokenizer (Tokenizer): Tokenizer.
        use_numpy (bool): Whether to use numpy arrays for X and y.
        label_map (dict): Mapping from label to index.
        data (list): List of raw data.
        X (list/np.ndarray): Preprocessed sentences.
        y (list/np.ndarray): Labels.
    """

    def __init__(self, data_path, verbose=1, tokenizer=None, use_numpy=True):
        """Initialize PubMedDataset.
        
        Args:
            data_path (str): Path to data.
            verbose (int): Verbosity level.
            tokenizer (Tokenizer): Tokenizer.
            use_numpy (bool): Whether to use numpy arrays for X and y.
        """
        self.data_path = data_path
        self.verbose = verbose
        self.tokenizer = tokenizer
        self.use_numpy = use_numpy
        self.label_map = {
            "BACKGROUND": 0,
            "OBJECTIVE": 1,
            "METHODS": 2,
            "RESULTS": 3,
            "CONCLUSIONS": 4
        }
        self.data = self._load_raw_data(self.data_path)
        if self.tokenizer is not None:
            self._tokenize_data(self.data)
        self.X = self._X()
        self.y = self._y()

    def _load_raw_data(self, data_path):
        with open(data_path, "r", encoding="utf8") as f:
            text = f.read()

        raw_data = []

        abstracts = text.split("###")[1:]  # Drop first line since it's empty.

        for abstract in tqdm.tqdm(abstracts,
                                  desc="Loading data",
                                  disable=self.verbose == 0):
            contents = [
                content for content in abstract.splitlines() if content
            ]  # Remove empty lines.
            abstract_id = contents[0]
            length = len(contents) - 1  # Remove abstract_id.
            for i, content in enumerate(contents[1:]):
                label_sentence_split = content.split("\t")
                label = self.label_map[label_sentence_split[0]]
                sentence = label_sentence_split[1]
                raw_data.append({
                    "abstract_id": abstract_id,
                    "abstract_index": i,
                    "abstract_length": length,
                    "label": label,
                    "sentence": sentence
                })

        return raw_data

    def _tokenize_data(self, data):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not specified.")

        tokenized_sentences = self.tokenizer.tokenize_batch(
            [item["sentence"] for item in data])
        for i, tokens in enumerate(tokenized_sentences):
            data[i]["tokens"] = tokens

    def _X(self):
        if self.tokenizer is None:
            X = [item["sentence"] for item in self.data]
            return np.array(X) if self.use_numpy else X
        else:
            X = [item["tokens"] for item in self.data]
            return np.array(X, dtype=object) if self.use_numpy else X

    def _y(self):
        y = [item["label"] for item in self.data]
        return np.array(y) if self.use_numpy else y


def load_data(data_path,
              train_file="train.txt",
              dev_file="dev.txt",
              test_file="test.txt",
              verbose=1,
              tokenizer=None,
              use_numpy=True,
              use_pickle=True):
    """Load train, dev, and test datasets.

    Args:
        data_path (str): Path to data.
        train_file (str, optional): Filename of training data.
            Defaults to "train.txt".
        dev_file (str, optional): Filename of dev data. Defaults to "dev.txt".
        test_file (str, optional): Filename of test data.
            Defaults to "test.txt".
        verbose (int, optional): Verbosity level. Defaults to 1.
        tokenizer (_type_, optional): Tokenizer used in data loading.
            Defaults to None.
        use_numpy (bool, optional): Whether to use numpy arrays.
            Defaults to True.
        use_pickle (bool, optional): Whether to load/dump datasets using
            pickle. Defaults to True.

    Returns:
        _type_: _description_
    """
    if use_pickle:
        # Check if pickle files exist and if so load those.
        suffix = tokenizer.name() if tokenizer is not None else "no_tokenizer"
        train_pickle_file = os.path.join(data_path, "train_" + suffix + ".pkl")
        dev_pickle_file = os.path.join(data_path, "dev_" + suffix + ".pkl")
        test_pickle_file = os.path.join(data_path, "test_" + suffix + ".pkl")
        if os.path.exists(train_pickle_file) and os.path.exists(
                dev_pickle_file) and os.path.exists(test_pickle_file):
            with open(train_pickle_file, "rb") as f:
                train_dataset = pickle.load(f)
            with open(dev_pickle_file, "rb") as f:
                dev_dataset = pickle.load(f)
            with open(test_pickle_file, "rb") as f:
                test_dataset = pickle.load(f)
        else:
            train_dataset, dev_dataset, test_dataset = load_data(
                data_path=data_path,
                train_file=train_file,
                dev_file=dev_file,
                verbose=verbose,
                tokenizer=tokenizer,
                use_numpy=use_numpy,
                use_pickle=False)
            # Pickle after.
            with open(train_pickle_file, "wb") as f:
                pickle.dump(train_dataset, f)
            with open(dev_pickle_file, "wb") as f:
                pickle.dump(dev_dataset, f)
            with open(test_pickle_file, "wb") as f:
                pickle.dump(test_dataset, f)
    else:
        train_dataset = PubMedDataset(data_path=os.path.join(
            data_path, train_file),
                                      verbose=verbose,
                                      tokenizer=tokenizer,
                                      use_numpy=use_numpy)
        dev_dataset = PubMedDataset(data_path=os.path.join(
            data_path, dev_file),
                                    verbose=verbose,
                                    tokenizer=tokenizer,
                                    use_numpy=use_numpy)
        test_dataset = PubMedDataset(data_path=os.path.join(
            data_path, test_file),
                                     verbose=verbose,
                                     tokenizer=tokenizer,
                                     use_numpy=use_numpy)
    return train_dataset, dev_dataset, test_dataset
