import json
import time
import argparse
import fasttext
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import f1_score, confusion_matrix

from model import ConLID

LANGS_MAP_UDHR_TO_GLOTC = {
    "src_Latn": "srd_Latn",
    "pes_Arab": "fas_Arab",
    "prs_Arab": "fas_Arab"
}
LANGS_UDHR_SKIP = ["dip_Latn", "tzm_Tfng", "pcd_Latn", "taj_Deva", "aii_Syrc", "guu_Latn", "mto_Latn", "chj_Latn", "auc_Latn"]

def compute_metrics(true_labels, pred_labels, dataset_labels):
    # Compute confusion matrix
    confusion_mat = confusion_matrix(true_labels, pred_labels)
    
    # Calculate TP, FP, FN, TN for each class
    FP = confusion_mat.sum(axis=0) - np.diag(confusion_mat)
    FN = confusion_mat.sum(axis=1) - np.diag(confusion_mat)
    TP = np.diag(confusion_mat)
    TN = confusion_mat.sum() - (FP + FN + TP)
    
    # Compute False Positive Rate per label
    fp_rate = FP / (FP + TN)
    # Handle division by zero by setting FPR to 0 where actual_negatives is 0
    fp_rate = np.nan_to_num(fp_rate, nan=0.0)
    
    # Compute F1 scores per label
    f1_scores = f1_score(true_labels, pred_labels, average=None, labels=dataset_labels)
    f1_micro = f1_score(true_labels, pred_labels, average='micro', labels=dataset_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro', labels=dataset_labels)
    
    # Compute FPR micro and macro
    fpr_micro = FP.sum() / (FP.sum() + TN.sum())
    fpr_macro = fp_rate.mean()
    
    # Get unique labels that appear in true_labels
    unique_labels = np.unique(true_labels)
    
    # Build the result dictionary
    result = {
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "fpr_micro": float(fpr_micro),
        "fpr_macro": float(fpr_macro),
    }
    
    # Add per-language metrics at the end
    per_language = {}
    for i, label in enumerate(unique_labels):
        per_language[str(label)] = {
            "f1": float(f1_scores[i]),
            "fpr": float(fp_rate[i])
        }
        
    return result, per_language

def compute_and_save_metrices(true_labels, pred_labels, dataset_labels, total_latency_ms, total_predictions, output_path):
    # Compute the f1/fpr metrics
    results, per_language = compute_metrics(true_labels=true_labels, pred_labels=pred_labels, dataset_labels=dataset_labels)

    # Add metadata
    results['total_latency_ms'] = total_latency_ms
    results['total_predictions'] = total_predictions
    results['num_languages'] = len(dataset_labels)
    results["per_language"] = per_language

    # Save the results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_top_k", type=int, default=10, help="Number of top predictions to consider for the ensembling method.")
    parser.add_argument("--checkpoints_dir", type=str, default='checkpoints', help="Path to the `checkpoints` directory.")
    parser.add_argument("--results_dir", type=str, default='results', help="Path to the `results` directory.")
    args = parser.parse_args()

    conlid_model = ConLID.from_pretrained(dir=f'{args.checkpoints_dir}/conlid')
    glotlid_model = fasttext.load_model(f"{args.checkpoints_dir}/glotlid/model.bin")

    # Get the common labels of both models
    conlid_labels = conlid_model.get_labels()
    glotlid_labels = [label.replace('__label__', '') for label in glotlid_model.labels]
    common_labels = list(set(conlid_labels) & set(glotlid_labels))

    # Load and filter the UDHR dataset
    dataset = load_dataset('cis-lmu/udhr-lid', split='test')
    dataset = dataset.map(
        lambda example: {'lang_id': LANGS_MAP_UDHR_TO_GLOTC.get(example['id'], example['id']), 'sentence': example['sentence']},
        remove_columns=['id', 'iso639-3', 'iso15924', 'language']
    )
    dataset = dataset.filter(lambda example: example['lang_id'] in common_labels and example['lang_id'] not in LANGS_UDHR_SKIP)
    dataset_labels = list(set(dataset['lang_id']))

    ### ConLID-S ###
    true_labels, pred_labels = [], []
    start_time = time.time()

    for example in tqdm(dataset, desc='Running for ConLID-S'):
        predictions, _ = conlid_model.predict(example['sentence'])
        true_labels.append(example['lang_id'])
        pred_labels.append(predictions[0])

    end_time = time.time()
    total_latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    compute_and_save_metrices(
        true_labels=true_labels, 
        pred_labels=pred_labels, 
        dataset_labels=dataset_labels, 
        total_latency_ms=total_latency_ms, 
        total_predictions=len(dataset),
        output_path=f"{args.results_dir}/udhr_conlid.json"
    )

    ### GlotLID-M ###
    true_labels, pred_labels = [], []
    start_time = time.time()

    for example in tqdm(dataset, desc='Running for GlotLID-M'):
        predictions, _ = glotlid_model.predict(example['sentence'])
        true_labels.append(example['lang_id'])
        pred_labels.append(predictions[0].replace('__label__', ''))

    end_time = time.time()
    total_latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    compute_and_save_metrices(
        true_labels=true_labels, 
        pred_labels=pred_labels, 
        dataset_labels=dataset_labels, 
        total_latency_ms=total_latency_ms, 
        total_predictions=len(dataset),
        output_path=f"{args.results_dir}/udhr_glotlid.json"
    )

    ### GlotLID-M + ConLID-S ###
    true_labels, pred_labels = [], []
    start_time = time.time()

    for example in tqdm(dataset, desc='Running for the ensembling method'):
        # Only get top 10-20 predictions instead of ALL predictions
        conlid_predictions, conlid_probabilities = conlid_model.predict(example['sentence'], k=args.ensemble_top_k)
        glotlid_predictions, glotlid_probabilities = glotlid_model.predict(example['sentence'], k=args.ensemble_top_k)
        glotlid_predictions = [label.replace('__label__', '') for label in glotlid_predictions]
        
        # Fast ensemble with limited candidates
        candidates = {}
        for label, prob in zip(conlid_predictions, conlid_probabilities):
            candidates[label] = candidates.get(label, 0.0) + prob
        for label, prob in zip(glotlid_predictions, glotlid_probabilities):
            candidates[label] = candidates.get(label, 0.0) + prob
        
        ensembled_prediction = max(candidates, key=candidates.get)
        
        true_labels.append(example['lang_id'])
        pred_labels.append(ensembled_prediction)

    end_time = time.time()
    total_latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    end_time = time.time()
    total_latency_ms = (end_time - start_time)

    compute_and_save_metrices(
        true_labels=true_labels, 
        pred_labels=pred_labels, 
        dataset_labels=dataset_labels, 
        total_latency_ms=total_latency_ms, 
        total_predictions=len(dataset),
        output_path=f"{args.results_dir}/udhr_ensemble_top_{args.ensemble_top_k}.json"
    )