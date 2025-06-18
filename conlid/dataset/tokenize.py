import os
import re
import argparse
import logging

from tqdm import tqdm
from functools import partial
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterator
from concurrent.futures import ProcessPoolExecutor

from sklearn.model_selection import train_test_split

from conlid.utils import save_txt, save_pickle

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def strip_strings(els: list[str]) -> list[str]:
    return [el.strip() for el in els if len(el.strip()) > 0]

def simple_span_tokenize(text: str, sents: list[str]) -> Iterator[tuple[int, int]]:
    start_index = 0
    for sent in sents:
        start_char = text.index(sent, start_index)
        end_char = start_char + len(sent)
        start_index = end_char
        yield start_char, end_char

class WordTokenizer(ABC):
    @abstractmethod
    def word_tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def sent_tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        pass

class SpaceTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__()

    def __split_words(self, text):
        """
        Split words on whitespace (space, newline, tab, vertical tab)
        and the control characters carriage return, formfeed, the null character,
        and zero width space characters.
        """
        pattern = r'[ \n\t\v\r\f\x00]+'
        words = re.split(pattern, text)
        return [word for word in words if word]

    def word_tokenize(self, text: str) -> list[str]:
        return strip_strings(self.__split_words(text))

    def sent_tokenize(self, text: str) -> list[str]:
        sentences = re.split(r'[.!?…;:—]+', text)
        sent = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 0]
        return strip_strings(sent)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        sents = self.sent_tokenize(text)
        return list(simple_span_tokenize(text, sents))

WORD_TOKENIZER_CACHE: dict[str, WordTokenizer] = {}

def load_word_tokenizer(language: str) -> WordTokenizer:
    """Load space tokenizer"""
    try:
        language.split("_")[1]
    except:
        raise ValueError(f"Language should be in the format 'LANG_SCRIPT', given '{language}'")
    
    if language not in WORD_TOKENIZER_CACHE:
        tokenizer = SpaceTokenizer()
            
        WORD_TOKENIZER_CACHE[language] = tokenizer

    return WORD_TOKENIZER_CACHE[language]

def split_glotlid(lang_id, datadir, test_ratio: int=0.15):
    """Splits the glotlid corpus into train/test"""
    lang_path = datadir / f"raw/{lang_id}"
    lang_datasets = os.listdir(lang_path)

    for lang_dataset in lang_datasets:

        if Path(datadir / f"train/raw/{lang_id}/{lang_dataset}").exists():
            continue

        lang_dataset_path = os.path.join(lang_path, lang_dataset)

        with open(lang_dataset_path, 'r') as file:
            raw_data = [line.strip() for line in file if len(line.strip()) > 0]

        if len(raw_data) > 10:
            # Split the data into train, and test sets
            test_size = min(int(test_ratio * len(raw_data)), 15000)
            train_data, test_data = train_test_split(raw_data, test_size=test_size, random_state=42)
            
            save_txt(train_data, datadir / f"train/raw/{lang_id}/{lang_dataset}")
            save_txt(test_data, datadir / f"test/raw/{lang_id}/{lang_dataset}")
        else:
            # Process without splitting
            save_txt(raw_data, datadir / f"train/raw/{lang_id}/{lang_dataset}")

def process_glotlid(lang_id, datadir):
    """tokenize the raw dataset assuming it is already split into train/test"""
    tokenizer = load_word_tokenizer(lang_id)
    
    for split in ['train', 'test']:
        lang_path = datadir / f"{split}/raw/{lang_id}"
        lang_datasets = os.listdir(lang_path)
    
        for lang_dataset in lang_datasets:
            lang_dataset_path = os.path.join(lang_path, lang_dataset)

            with open(lang_dataset_path, 'r') as file:
                raw_data = [line.strip() for line in file]

            # Process without splitting
            tokenized_train_data = [tokenizer.word_tokenize(line) for line in raw_data]
            save_pickle(tokenized_train_data, datadir / f"{split}/tokenized/{lang_id}/{lang_dataset.replace('.txt','')}.pkl")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="data/glotlid/v3.1", help="Path to GlotLID-C directory")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of workers to process in parallel")
    args = parser.parse_args()

    logging.info(f"Number of workers: {args.num_workers}")

    datadir = Path(args.datadir)
    lang_ids = os.listdir(datadir / 'v3.1')

    # split in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        split_fn = partial(split_glotlid, datadir=datadir)
        list(tqdm(executor.map(split_fn, lang_ids), total=len(lang_ids)))

    # tokenize in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        process_fn = partial(process_glotlid, datadir=datadir)
        list(tqdm(executor.map(process_fn, lang_ids), total=len(lang_ids)))