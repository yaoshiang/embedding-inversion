"""This script runs deep dreams / adversarial attack on an E5 model."""

import evaluate
import torch
import transformers
from datasets import concatenate_datasets, load_dataset

from . import attack_e5, e5_utils


def download_and_evaluate() -> None:
    """Loads and evaluates how effective the attack is on various MTEB text retrieval datasets."""
    # List of datasets
    # Handpicked as more common datasets
    dataset_paths = [
        "mteb/arguana",
        "mteb/climate-fever",
        "mteb/dbpedia",
        "mteb/fever",
        "mteb/fiqa",
        "mteb/hotpotqa",
        "mteb/msmarco",
        "mteb/nfcorpus",
        "mteb/nq",
        "mteb/scidocs",
        "mteb/scifact",
        "mteb/touche2020",
        "mteb/trec-covid",
    ]
    word_overlap_scores = {}

    model = transformers.AutoModel.from_pretrained("intfloat/e5-small")
    for path in dataset_paths:
        try:
            queries_dataset = load_dataset(path, "queries", split="queries[:10]")
            corpus_dataset = load_dataset(path, "corpus", split="corpus[:10]")

            # Add "passage" and "query" prefixes to the input text
            queries_dataset = queries_dataset.map(lambda dict: {"text": "query: " + dict["text"]})
            corpus_dataset = corpus_dataset.map(lambda dict: {"text": "passage: " + dict["text"]})

            # Combine datasets
            combined_dataset = concatenate_datasets([queries_dataset, corpus_dataset])

            input_texts = combined_dataset["text"]

            # ---Assert---

            # Assert that the input text begins with either "query" or "passage"
            assert all(
                text.startswith("query: ") for text in combined_dataset[:10]["text"]
            ), "Not every query text starts with 'query: "
            assert all(
                text.startswith("passage: ") for text in combined_dataset[10:]["text"]
            ), "Not every corpus text starts with 'passage: '"

            embeddings, batch_dict = e5_utils.run_e5(model, input_texts)

            assert isinstance(embeddings, torch.Tensor)
            assert embeddings.dim() == 2, embeddings.dim()
            assert embeddings.shape[1] <= model.config.max_position_embeddings

            # Generate predictions
            predictions = attack_e5.attack_e5_small(embeddings, batch_dict)

            bleu = evaluate.load("bleu")
            rouge = evaluate.load("rouge")

            references = [[ref] for ref in input_texts]

            # Assert - Check if predictions and references are of the same length
            assert len(predictions) == len(references), "Number of predictions does not match number of references."

            bleu_score = bleu.compute(predictions=predictions, references=references)
            rouge_score = rouge.compute(predictions=predictions, references=references)

            # Rudimentary metric measuring how many of the same words do the input texts and predictions share
            word_overlap_score = calculate_word_overlap_score(predictions=predictions, references=references)
            word_overlap_scores[path] = word_overlap_score

            print(f"Dataset: {path}")
            print(f"BLEU score: {bleu_score}")
            print(f"ROUGE score: {rouge_score}")
            print(f"Overlap score: {word_overlap_score}")
            print()
        except Exception as ex:
            print(f"An error occurred with dataset {path}. Error {ex}")
            word_overlap_scores[path] = None

    # Print the final word overlap scores for each dataset
    print("Final word overlap scores:")
    for dataset, score in word_overlap_scores.items():
        print(f"{dataset}: {score}")


def calculate_word_overlap_score(predictions, references) -> float:
    """Calculates the average word overlap between the predictions and the input text.

    Args:
        predictions (list of str): List containing predicted sentences as strings.
        references (list of list of str): List containing reference sentences where each reference is a list of strings.

    Returns:
        float: the average word overlap score.
    """
    total_word_overlap = 0
    total_input_words = 0

    if len(references) == 0 or len(predictions) == 0:
        return 0.0

    for ref, pred in zip(references, predictions):
        ref_words = ref[0].split()
        pred_words = pred.split()
        total_word_overlap += len(set(ref_words) & set(pred_words))
        total_input_words += len(set(ref_words))

    word_overlap_score = total_word_overlap / total_input_words
    return word_overlap_score


if __name__ == "__main__":
    download_and_evaluate()
