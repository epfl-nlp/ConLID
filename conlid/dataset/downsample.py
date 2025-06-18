import os
import logging
import random
import argparse

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from conlid.utils import load_pickle, save_pickle

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/glotlid/v3.1", help="Path to GlotLID-C directory")
    parser.add_argument("--cap_size", type=int, default=100000, help="Maximum datasize")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    # count the train data size
    label_count = defaultdict(int)

    for lang_id in tqdm(os.listdir(data_dir / f'train/raw'), desc=f"Counting train data size"):
        lang_path = data_dir / f"train/raw/{lang_id}"
        lang_datasets = os.listdir(lang_path)

        for lang_dataset in lang_datasets:
            lang_dataset_path = os.path.join(lang_path, lang_dataset)

            with open(lang_dataset_path, 'r') as file:
                raw_data = [line.strip() for line in file if len(line.strip()) > 0]
            
            label_count[lang_id] += len(raw_data)

    label_count = dict(label_count)
    langs = list(label_count.keys())

    # downsample to 100K
    logging.info(f'Capping to {args.cap_size}')
    OUTPUT_PATH = data_dir / "train/tokenized_downsampled"
    high_resource_langs_count = {label: size for label, size in label_count.items() if size > args.cap_size}

    for lang in tqdm(langs, desc='Capping high resource languages'):
        new_size = 0

        if lang in high_resource_langs_count:
            lang_ratio_to_keep = args.cap_size / high_resource_langs_count[lang]

        lang_path = data_dir / f"train/tokenized/{lang}"
        datasets = sorted(os.listdir(lang_path))

        for dataset in datasets:
            dataset_path = os.path.join(lang_path, dataset)
            doc_list = load_pickle(dataset_path)

            # downsample high-resource languages per dataset
            if lang in high_resource_langs_count:
                lang_size_to_keep = round(len(doc_list) * lang_ratio_to_keep)
                doc_list = random.sample(doc_list, lang_size_to_keep)
                new_size += len(doc_list)

            save_pickle(doc_list, Path(os.path.join(OUTPUT_PATH, lang, dataset)))

        if lang in high_resource_langs_count:
            logging.info(f'{lang}: {high_resource_langs_count[lang]} -> {new_size}')