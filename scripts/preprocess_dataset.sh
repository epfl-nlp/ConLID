#!/bin/bash

# Generate synthetic dataset for 160 UND labels
# Generating 120000 sentences per label, so that we will have ~100K after split
python -m conlid.dataset.generate_und \
    --data_dir $REPO_PATH/data/glotlid/v3.1 \
    --num_sentences 120000

# Split the dataset into train(85%) / test(15%); Save under `[split]/raw`
# Tokenize the dataset (on space); Saved under `[split]/tokenized_space`
python -m conlid.dataset.tokenize \
    --data_dir $REPO_PATH/data/glotlid/v3.1 \
    --num_workers $(nproc)

# Downsample high resource langauges to 100K
python -m conlid.dataset.downsample \
    --data_dir $REPO_PATH/data/glotlid/v3.1 \
    --cap_size 100000

# Converts the train/test dataset into word/character-level ngrams as in Fasttext; Saved as HF dataset
python -m conlid.dataset.convert_to_ngram \
    --data_dir $REPO_PATH/data/glotlid/v3.1 \
    --config_path $REPO_PATH/data/conlid/config.json \
    --num_workers $(nproc)

