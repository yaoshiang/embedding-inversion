from . import attack_e5
from . import dataprep

if __name__ == "__main__":
    MAX_EXAMPLES = 100

    embeddings, batch_dict = dataprep.create_hotpotqa_dataset_for_e5(max_examples=MAX_EXAMPLES)
    attack_e5.attack_e5_small(embeddings, batch_dict)
