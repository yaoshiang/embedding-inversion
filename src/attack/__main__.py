from . import attack_e5

if __name__ == "__main__":
    embeddings, batch_dict = attack_e5.create_sample_dataset()
    attack_e5.attack_e5_small(embeddings, batch_dict)
