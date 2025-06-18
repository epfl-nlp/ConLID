import os
import logging
import argparse

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder

from datasets import load_from_disk, Dataset, DatasetDict

from conlid.utils import load_json, load_pickle, save_pickle

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def hash(string: str) -> int:
    """The same hash function as used in fasttext"""
    h = 2166136261
    for char in string:
        # Get the UTF-8 encoded bytes of the character
        for byte in char.encode('utf-8'):
            # Convert byte to signed 8-bit integer (-128 to 127 range)
            byte_value = byte if byte < 128 else byte - 256
            h = h ^ byte_value
            h = h * 16777619
    return h & 0xFFFFFFFF

def generate_ngrams(string: str, minn, maxn) -> list:
    ngrams = set()
    length = len(string)

    for n in range(minn, maxn + 1):
        for i in range(length - n + 1):
            ngram = string[i:i + n]
            ngrams.add(ngram)

    return list(ngrams)

def build_vocab(langs, pad_id, unk_id, ngrams_dir, min_count, train_dir):
    """Builds a vocabulary of words"""
    word2id_path = ngrams_dir / "word2id.pkl"

    if word2id_path.exists():
        logging.info(f'Loaded word2id from {word2id_path}')
        return load_pickle(word2id_path)

    # get path to all docs
    train_doc_paths = [] # (lang_id, dataset_path)
    for lang in langs:
        lang_path = train_dir / lang
        datasets = sorted(os.listdir(lang_path))

        for dataset in datasets:

            dataset_path = lang_path / dataset
            train_doc_paths.append((lang, dataset_path))
    
    train_doc_paths.sort()
    
    # count words and generate ngrams
    word2count = defaultdict(int)
    for _, doc_path in tqdm(train_doc_paths, desc="Counting words"):
        doc = load_pickle(doc_path)

        for sent in doc:
            for word in sent:
                word2count[word] += 1

    n_total_word = sum(word2count.values())
    logging.info(f'Number of total words: {n_total_word}')
        
    words = [word for word, count in word2count.items() if count >= min_count]

    # building vocabulary
    word2id = {}
    word2id['<pad>'] = pad_id
    word2id['<unk>'] = unk_id

    index = 2  # 0 for padding, 1 for unk
    for word in tqdm(words, desc="Hashing tokens"):
        word2id[word] = index
        index += 1
    logging.info(f'Number of used words: {len(word2id)}')
    
    save_pickle(word2id, word2id_path)
    return word2id

def encode_labels(langs, ngrams_dir):
    """Encodes the class labels."""
    label2id_path = ngrams_dir / "label2id.pkl"

    if label2id_path.exists():
        logging.info(f'Loaded label2id from {label2id_path}')
        return load_pickle(label2id_path)

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(langs)
    label2id = {label: encoded_label for label, encoded_label in zip(label_encoder.classes_, train_labels)} # {label: id}

    save_pickle(label2id, ngrams_dir / "label2id.pkl")
    return label2id

def read_sentences(examples):
    """Reads tokenized docs as lists."""
    output = {
        'label': [],
        'doc': [],
        'tokens': []
    }
    
    for label, doc_path in zip(examples['label'], examples['doc_path']):
        doc = load_pickle(doc_path)

        if len(doc) > 0:
            doc_name = doc_path.replace('.pkl','').split('/')[-1]

            # skip empty examples
            doc = [example for example in doc if len(example) > 0]

            output['label'].extend([label] * len(doc))
            output['doc'].extend([doc_name] * len(doc))
            output['tokens'].extend(doc)
    
    return output

def convert_tokens_to_hf_dataset(langs, ngrams_dir, train_dir, test_dir):
    """Reads the tokenized dataset from .pkl and converts it into HF dataset format"""
    out_data_path = ngrams_dir / 'tokenized'
    
    if out_data_path.exists():
        return
    
    # test data
    test_doc_paths = [] # (lang_id, dataset_path)

    for lang in langs:
        lang_path = train_dir / lang
        datasets = sorted(os.listdir(lang_path))

        for dataset in datasets:
            dataset_path = lang_path / dataset
            test_doc_paths.append((lang, dataset_path))

    test_doc_paths.sort()
    print(f'# test docs: {len(test_doc_paths)}')

    test_ds_paths = Dataset.from_dict({
        'label': [label for label, _ in test_doc_paths],
        'doc_path': [path for _, path in test_doc_paths]
    })

    test_ds = test_ds_paths.map(
        read_sentences,
        batched=True,
        batch_size=1,
        desc="Processing documents",
        remove_columns=test_ds_paths.column_names,
        drop_last_batch=False,
    )

    # train data
    train_doc_paths = [] # (lang_id, dataset_path)

    for lang in langs:

        lang_path = test_dir / lang
        datasets = sorted(os.listdir(lang_path))

        for dataset in datasets:
            dataset_path = lang_path / dataset
            train_doc_paths.append((lang, dataset_path))

    train_doc_paths.sort()
    print(f'# train docs: {len(train_doc_paths)}')

    train_ds_paths = Dataset.from_dict({
        'label': [label for label, _ in train_doc_paths],
        'doc_path': [path for _, path in train_doc_paths]
    })

    train_ds = train_ds_paths.map(
        read_sentences,
        batched=True,
        batch_size=1,
        desc="Processing documents",
        remove_columns=train_ds_paths.column_names,
        drop_last_batch=False,
    )

    # Save to disk
    dataset = DatasetDict({
        'train': train_ds,
        'test': test_ds,
    }).save_to_disk(str(out_data_path))

def convert_tokens_to_ngrams(examples, label2id, word2id, n_words, add_ngram, bucket, unk_id, minn, maxn):
    hashids = []    
    
    for tokens in examples['tokens']:
        sent_hashes = []
        for word in tokens:
            word_hashes = []
            word_hash = word2id.get(word)
            
            if word_hash is not None:
                word_hashes.append(word_hash)
    
            if add_ngram:
                ngrams = generate_ngrams(f"<{word}>", minn, maxn)
                word_hashes += [n_words + hash(ngram) % bucket for ngram in ngrams]
    
            if word_hashes:
                sent_hashes.extend(word_hashes)
    
        if len(sent_hashes) == 0:
            logging.info(f'No word representation found for {tokens}\nSetting to [UNK_ID]')
            sent_hashes = [unk_id]

        hashids.append(sent_hashes)

    label_id = [label2id[label] for label in examples['label']]

    return {"hashids": hashids, "label_id": label_id}

def process_test_dataset(langs, ngrams_dir, add_ngram, bucket, unk_id, minn, maxn, batch_size=16, num_proc=32):
    """
    Converts tokens into ngrams for test datasets.
    """

    # Ensure output directory exists
    input_dir = ngrams_dir / f'tokenized/test'
    output_dir = ngrams_dir / f'hashids/test'

    if output_dir.exists():
        return

    label2id = encode_labels(langs)
    word2id = build_vocab(langs)
    n_words = len(word2id)

    dataset = load_from_disk(str(input_dir))
    logging.info(f"Loaded dataset from {input_dir}")

    # filter languages with bible datasets only
    dataset = dataset.filter(lambda sample: sample['label'] in label2id)
    logging.info(dataset)

    dataset = dataset.map(
        convert_tokens_to_ngrams,
        batched=True,
        batch_size=batch_size,
        remove_columns=['tokens'],
        drop_last_batch=False,
        num_proc=num_proc,
        desc=f"Processing test",
        fn_kwargs={'label2id': label2id, 'word2id': word2id, 'n_words': n_words, 'add_ngram': add_ngram, 'bucket': bucket, 'unk_id': unk_id, 'minn': minn, 'maxn': maxn}
    )

    dataset.save_to_disk(str(output_dir))
    logging.info(f"Saved dataset to {output_dir}")

def process_train_dataset_by_shards(langs, ngrams_dir, add_ngram, bucket, unk_id, minn, maxn, batch_size=16, num_proc=32):
    """
    Converts tokens into ngrams for test datasets.
    Process a large dataset shard by shard, saving each processed shard to disk.
    """

    # Ensure output directory exists
    input_dir = ngrams_dir / f'tokenized/train'
    output_shards = ngrams_dir / f'hashids/train_shards'
    output_shards.mkdir(parents=True, exist_ok=True)

    file_names = [cache_file['filename'] for cache_file in load_from_disk(str(input_dir)).cache_files]
    file_names.sort()

    label2id = encode_labels(langs)
    word2id = build_vocab(langs)
    n_words = len(word2id)
    n_shards = len(file_names)
    logging.info(f"Dataset has {n_shards} shards")

    # Process and save each shard separately
    for shard_idx, file_name in enumerate(file_names):
        shard_output_path = output_shards / f'processed_shard_{shard_idx}'
        
        if shard_output_path.exists():
            logging.info(f"Shard {shard_idx + 1}/{n_shards} already exists")
            continue

        logging.info(f"Processing shard {shard_idx + 1}/{n_shards}")

        # Load single shard
        shard = Dataset.from_file(file_name)
    
        # Process the shard
        shard = shard.map(
            convert_tokens_to_ngrams,
            batched=True,
            batch_size=batch_size,
            drop_last_batch=False,
            num_proc=num_proc,
            remove_columns=['tokens'],
            desc=f"Processing shard {shard_idx + 1}/{n_shards}",
            fn_kwargs={'label2id': label2id, 'word2id': word2id, 'n_words': n_words, 'add_ngram': add_ngram, 'bucket': bucket, 'unk_id': unk_id, 'minn': minn, 'maxn': maxn}
        )

        # Save each processed shard to disk
        shard.save_to_disk(str(shard_output_path))

    # Load all processed shards and concatenate them into one dataset
    logging.info("Save all processed shard")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/glotlid/v3.1", help="Path to GlotLID-C directory")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of workers to process in parallel")
    parser.add_argument("--config_path", type=str, default="conlid/config.json", help="Path to the `config.json` file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train/tokenized_downsampled"
    test_dir = data_dir / "test/tokenized"
    ngrams_dir = data_dir / "ngrams"

    config = load_json(Path(args.config_path))

    # hypterparameters for tokenization
    add_ngram = config['bucket'] > 0 and config['maxn'] > 0

    langs = sorted(os.listdir(train_dir))
    logging.info(f'Number of languages being used: {len(langs)}')

    build_vocab(langs=langs, pad_id=config['pad_id'], unk_id=config['unk_id'], ngrams_dir=ngrams_dir, train_dir=train_dir)
    encode_labels(langs=langs, ngrams_dir=ngrams_dir)
    
    # convert into HF dataset for more efficient training
    convert_tokens_to_hf_dataset(langs=langs, ngrams_dir=ngrams_dir, train_dir=train_dir, test_dir=test_dir)

    # Generate character and word level ngrams
    process_test_dataset(langs=langs, ngrams_dir=ngrams_dir, add_ngram=add_ngram, bucket=config['bucket'], unk_id=config['unk_id'], minn=config['minn'], maxn=config['maxn'], num_proc=args.num_workers)
    process_train_dataset_by_shards(langs=langs, ngrams_dir=ngrams_dir, add_ngram=add_ngram, bucket=config['bucket'], unk_id=config['unk_id'], minn=config['minn'], maxn=config['maxn'], num_proc=args.num_workers)