import os
import logging
import pathlib

from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch
from torch.nn.utils.rnn import pad_sequence

from datasets import load_from_disk, concatenate_datasets
from transformers import TrainingArguments, Trainer, set_seed, HfArgumentParser, TrainingArguments

from conlid.train.models import ConLID_H, ConLID_S, LID_CE, LID_SCL

from sklearn.metrics import f1_score
import numpy as np

from conlid.utils import load_json, load_pickle

MODELS_MAP = {
    'lid_ce': LID_CE,
    'lid_scl': LID_SCL,
    'conlid_s': ConLID_S,
    'conlid_h': ConLID_H
}

N_GPUS = int(os.environ["WORLD_SIZE"])

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)

@dataclass
class ConLIDScriptArguments:
    """
    Script arguments for the ConLID training script.
    """
    model_type: str = field(
        default='conlid_s',
        metadata={"help": "Should be one of [lid_ce, lid_scl, conlid_s]"},
    )
    data_dir: str = field(
        default="data/glotlid/v3.1",
        metadata={"help": "Path to GlotLID-C directory"},
    )
    config_path: str = field(
        default="conlid/config.json",
        metadata={"help": "Path to the `config.json` file"},
    )
    contrastive_temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature parameter for contrastive loss"},
    )
    bank_size: int = field(
        default=2048,
        metadata={"help": "Size of the memory bank (`M` from the paper)"},
    )
    min_neg: int = field(
        default=1024,
        metadata={"help": "Minimum number of negatives for hard selection (`K` from the paper)"},
    )

def create_datasets(train_shards_dir, test_dir, sample_test_ratio = 0.1):
    """Creates train and test Iterable Datasets."""
    # Load train dataset
    train_dataset = concatenate_datasets([load_from_disk(str(train_shards_dir / shard_path)) for shard_path in os.listdir(train_shards_dir)])
    train_dataset = train_dataset.select_columns(['hashids', 'label_id']).rename_column("label_id","labels").rename_column("hashids","input_ids")

    # Load test datasets
    test_glotlid_dataset = load_from_disk(str(test_dir))
    test_glotlid_dataset = test_glotlid_dataset.select_columns(['hashids', 'label_id']).rename_column("label_id","labels").rename_column("hashids","input_ids")

    # sample `sample_test_ratio` % of the `test_glotlid_dataset`
    test_glotlid_dataset = test_glotlid_dataset.select(range(int(sample_test_ratio * len(test_glotlid_dataset))))

    test_dataset = {
        'glotc_test': test_glotlid_dataset
    }

    return train_dataset, test_dataset

def train(script_args, training_args):
    set_seed(training_args.seed)

    # set up the paths
    data_dir = Path(script_args.data_dir)
    ngrams_dir = data_dir / 'ngrams'
    train_shards_dir = ngrams_dir / 'hashids/train_shards'
    test_dir = ngrams_dir / 'hashids/test'

    # load vocabulary and labels
    word2id = load_pickle(ngrams_dir / "word2id.pkl")
    logger.info(f"Number of words as features: {len(word2id)}")

    label2id = load_pickle(ngrams_dir / "label2id.pkl") # {label: index}
    id2label = {v: k for k, v in label2id.items()} # {index: label}
    num_classes = len(id2label)
    logger.info(f"Number of labels: {num_classes}")

    # load the config file
    config = load_json(Path(script_args.config_path))
    add_ngram = config['bucket'] > 0 and config['maxn'] > 0

    if add_ngram:
        vocab_size = len(word2id) + config['bucket']
    else:
        vocab_size = len(word2id)
    logger.info(f"Vocab Size: {vocab_size}")

    def collate_fn(examples):
        """pads to the max length in the batch"""
        # inputs and labels
        inputs_padded = pad_sequence(
            [torch.tensor(example['input_ids']) for example in examples],
            batch_first=True,
            padding_value=config['pad_id']
        ) # [batch_size x n_samples]
        labels = torch.tensor([example['labels'] for example in examples])

        return {"input_ids": inputs_padded, "labels": labels}

    def compute_metrics(pred):
        """calculates weighted f1 score"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1) # pred.predictions = logits

        f1 = f1_score(labels, preds, average='weighted')
        return {'f1': f1}

    # Initialized the datasets and the models
    train_dataset, eval_dataset = create_datasets(train_shards_dir=train_shards_dir, test_dir=test_dir, sample_test_ratio=0.1)
    model = MODELS_MAP[script_args.model_type](
        vocab_size=vocab_size,
        num_classes=num_classes,
        **config,
        **asdict(script_args)
    )
    
    # Resume the wandb log
    os.environ["WANDB_RUN_ID"] = training_args.run_name
    os.environ["WANDB_RESUME"] = "allow"

    # Log `contrastive_loss` in wandb for the trainings with contrastive loss
    class ContrastiveTrainer(Trainer):
        def log(self, logs):
            # Retrieve the last computed losses from the model
            if self.model.ce_loss_log and self.model.contrastive_loss_log and ('loss' in logs):
                logs['ce_loss'] = np.mean(self.model.ce_loss_log)
                logs['contrastive_loss'] = np.mean(self.model.contrastive_loss_log)
                self.model.reset_loss_log()
            elif 'loss' not in logs:
                self.model.reset_loss_log()
            
            super().log(logs)

    if script_args.model_type == 'lid_ce':
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            compute_metrics=compute_metrics
        )
    else:
        trainer = ContrastiveTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            compute_metrics=compute_metrics
        )

    # Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save
    torch.cuda.synchronize()
    trainer.save_pretrained(str(training_args.output_dir / "final"))
    logger.info(f"Saved at: {training_args.output_dir / 'final'}")

if __name__=="__main__":
    parser = HfArgumentParser((ConLIDScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    train(script_args=script_args, training_args=training_args)
