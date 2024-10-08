from . import attack_e5, dataprep

if __name__ == "__main__":
    MAX_EXAMPLES = 2

    texts, reconstructed_texts, batch_dict, embeddings = dataprep.create_hotpotqa_dataset_for_e5(
        max_examples=MAX_EXAMPLES, max_length=512
    )

    pred = attack_e5.attack_e5_small(embeddings, batch_dict)
