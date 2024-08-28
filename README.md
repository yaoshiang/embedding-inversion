# embedding-inversion

This repo shows how to run DeepDreams style attacks on 
an embedding model to attempt to recover input strings.

We study the Microsoft E5 model given it is based on the
well understood BERT model. 

https://www.microsoft.com/en-us/research/publication/text-embeddings-by-weakly-supervised-contrastive-pre-training/

## python tools/print_model.py

This script prints intermediate datastructure from BERT. 

## python src/tools/bert_priors.py

This script runs BERT with masks to probe its priors. 

## python src/tools/distilbert.py

This script outputs multiple intermediate datastructures from DistilBERT and it's tokenizers.

## Attack script

pip install -e .
python -m attack
