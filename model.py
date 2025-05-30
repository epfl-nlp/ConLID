import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_model as load_model_as_safetensor


class SpaceTokenizer:
    def __init__(self):
        super().__init__()

    def _strip_strings(self, tokens: list[str]) -> list[str]:
        return [token.strip() for token in tokens if token.strip()]

    def _split_words(self, text: str) -> list[str]:
        pattern = r'[\n\t\v\r\f\x00 ]+'
        return [word for word in re.split(pattern, text) if word]

    def word_tokenize(self, text: str) -> list[str]:
        """
        Tokenizes input text by whitespace (space, newline, tab, vertical tab),
        the control characters carriage return, formfeed, the null character, and zero width space characters.

        Args:
            text (str): Input string.

        Returns:
            list[str]: Tokenized words.
        """
        return self._strip_strings(self._split_words(text))


class ConLID(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        num_classes: int,
        bucket: int,
        min_count: int,
        minn: int,
        maxn: int,
        aggr: str,
        pad_id: int = 0,
        unk_id: int = 1
    ):
        """
        Initializes the ConLID model.

        Args:
            vocab_size (int): Vocabulary size.
            embedding_size (int): Size of word embeddings.
            num_classes (int): Number of languages.
            bucket (int): Number of hash buckets for n-grams.
            min_count (int): Minimum frequency threshold for words.
            minn (int): Minimum n-gram length.
            maxn (int): Maximum n-gram length.
            aggr (str): Aggregation strategy of word embeddings in a sentence: 'mean', 'max', or 'sum'.
            pad_id (int): Padding token ID.
            unk_id (int): Unknown token ID.
        """
        super().__init__()
        assert aggr in {'mean', 'max', 'sum'}

        self.aggr_fn = self._get_aggregation_fn(aggr)

        self.vocab = None
        self.id2label = None
        self.convert_id2label = None

        self.bucket = bucket
        self.min_count = min_count
        self.minn = minn
        self.maxn = maxn
        self.add_ngram = bucket > 0 and maxn > 0
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.vocab_size = vocab_size

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = SpaceTokenizer()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)

    def _get_aggregation_fn(self, aggr: str):
        """
        Returns a callable aggregation function based on strategy.
        """
        if aggr == 'mean':
            return lambda x, mask: torch.sum(x * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1, keepdim=True)
        elif aggr == 'max':
            return lambda x, mask: torch.max(x * mask.unsqueeze(-1) + (mask.unsqueeze(-1) - 1) * float('-inf'), dim=1)[0]
        else:
            return lambda x, mask: torch.sum(x * mask.unsqueeze(-1), dim=1)

    @staticmethod
    def _load_json(path: str):
        """
        Loads a JSON file.
        """
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _hash(self, string: str) -> int:
        """
        FNV-1a hashing algorithm for n-grams. Used by Fasttext.

        Args:
            string (str): String to hash.

        Returns:
            int: Hash value.
        """
        h = 2166136261
        for char in string:
            for byte in char.encode('utf-8'):
                byte_value = byte if byte < 128 else byte - 256
                h ^= byte_value
                h *= 16777619
        return h & 0xFFFFFFFF

    def _generate_ngrams(self, word: str) -> list[str]:
        """
        Generates character-level n-grams from a word, using window size of `minn` to `maxn`.

        Args:
            word (str): Input word.

        Returns:
            list[str]: List of n-grams.
        """
        return [word[i:i+n] for n in range(self.minn, self.maxn + 1) for i in range(len(word) - n + 1)]

    def _tokenize(self, text: str):
        """
        Tokenizes the given text on space.
        """
        return self.tokenizer.word_tokenize(text)
    
    def _tokens2ngrams(self, tokens: list[str]) -> list[int]:
        """
        Converts tokens to a list of token and n-gram hash ids.

        Args:
            tokens (list[str]): Tokenized input.

        Returns:
            list[int]: List of token/n-gram ids.
        """
        ids = []
        for word in tokens:
            if word in self.vocab:
                ids.append(self.vocab[word])

            if self.add_ngram:
                # <word> is used by Fasttext
                for ngram in self._generate_ngrams(f"<{word}>"):
                    hashed = self._hash(ngram)
                    ids.append(self.vocab_size + hashed % self.bucket)

        return ids

    def _compute_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Computes sentence-level embeddings. Masks the `unk` and `pad` tokens.

        Args:
            ids (torch.Tensor): Padded batch of token ids.

        Returns:
            torch.Tensor: Aggregated sentence embeddings.
        """
        mask = ~((ids == self.unk_id) | (ids == self.pad_id))
        embeddings = self.embedding(ids)
        return self.aggr_fn(embeddings, mask).detach()

    @classmethod
    def from_pretrained(cls, dir: str):
        """
        Loads model from the given directory.
        Directory must contain the following files: `model.safetensors`, `config.json`, `vocab.json`, `labels.json`

        Args:
            dir (str): Path to the directory containing the model files.

        Returns:
            ConLID: Loaded model.
        """
        config = cls._load_json(os.path.join(dir, 'config.json'))
        model = cls(**config)

        model_path = os.path.join(dir, 'model.safetensors')
        load_model_as_safetensor(model, model_path, device=model.device)
        model.to(model.device)

        model.vocab = cls._load_json(os.path.join(dir, 'vocab.json'))
        model.vocab_size = len(model.vocab)

        labels = cls._load_json(os.path.join(dir, 'labels.json'))
        model.id2label = {i: label for label, i in labels.items()}
        model.convert_id2label = np.vectorize(model.id2label.get)

        return model

    def get_labels(self) -> list[str]:
        """
        Returns the model's labels.

        Returns:
            list[str]: List of label names.
        """
        return list(self.id2label.values())

    def predict(self, text: str, k: int = 1) -> tuple[list[str], list[float]]:
        """
        Predicts the top-k labels for a given text.

        Args:
            text (str): Input text.
            k (int): Number of top predictions to return.

        Returns:
            tuple[list[str], list[float]]: Predicted labels and their probabilities.
        """
        assert k >= 1, f'`k` must be >= 1; got k={k}'

        tokens = self._tokenize(text)
        ids = self._tokens2ngrams(tokens)
        input_tensor = torch.tensor(ids, dtype=torch.int, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self(input_tensor)
            probs = logits.softmax(dim=-1).squeeze()

        top_probs, top_indices = probs.topk(k)
        predictions = [self.id2label[i.item()] for i in top_indices]
        probabilities = top_probs.tolist()

        return predictions, probabilities

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Batch of input token ids.

        Returns:
            torch.Tensor: Logits for each class.
        """
        embeddings = self._compute_embeddings(input_ids)
        return self.fc(embeddings)