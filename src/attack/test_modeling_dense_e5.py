import torch
from transformers import AutoModel, AutoTokenizer, BertModel

from .e5_utils import run_e5, run_one_hot_e5
from .modeling_dense_e5 import DenseBertEmbeddings, DenseE5


def test_one_hot_bert_embeddings_initialization():  # noqa: D103
    # ---Arrange---

    # Load a pre-trained BERT model to use its embeddings
    model = BertModel.from_pretrained("intfloat/e5-small")  # Not 'bert-base-uncased'
    bert_embeddings = model.embeddings

    # ---Act---

    # Instantiate custom embeddings class
    one_hot_bert_embeddings = DenseBertEmbeddings(bert_embeddings)

    # ---Assert---

    # Assertions to ensure everything was copied/transferred correctly
    assert torch.all(
        one_hot_bert_embeddings.position_ids == torch.arange(model.config.max_position_embeddings).expand((1, -1))
    ), "Position IDs not initialized correctly"

    # Ensure that other components are copied correctly
    assert torch.all(
        torch.eq(one_hot_bert_embeddings.token_type_embeddings.weight, bert_embeddings.token_type_embeddings.weight)
    ), "Token type embeddings not copied correctly"
    assert torch.all(
        torch.eq(one_hot_bert_embeddings.LayerNorm.weight, bert_embeddings.LayerNorm.weight)
    ), "LayerNorm weights not copied correctly"
    assert one_hot_bert_embeddings.dropout.p == bert_embeddings.dropout.p, "Dropout probability mismatch"


# Define a test function for forward pass
def test_sparse_bert_embeddings_forward_pass():
    # ---Arrange---

    # Load a pre-trained BERT model
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_embedding = bert_model.embeddings
    del bert_model

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Define a sample input text
    input_text = "query: how much protein should a female eat"

    # Tokenize the input text to get token IDs
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # ---Act---

    # Get embeddings from the original BERT model
    expected = bert_embedding(input_ids).detach()

    # Instantiate custom embeddings class
    one_hot_bert_embeddings = DenseBertEmbeddings(bert_embedding)

    # Convert token IDs to one-hot representations for your custom embeddings
    one_hot_vectors = torch.nn.functional.one_hot(input_ids, num_classes=tokenizer.vocab_size).float()

    # Get embeddings from the custom one-hot embeddings
    actual = one_hot_bert_embeddings(one_hot_vectors).detach()

    # ---Assert---

    # Check the shapes of the embeddings to ensure consistency
    assert expected.shape == actual.shape, "Embedding shapes do not match"

    # Further checks can be added based on your specific requirements for the custom embedding layer
    # For example, you can check the values to ensure they're within expected ranges or have similar magnitudes
    assert expected.mean() == actual.mean(), "Embedding mean mismatch"
    assert expected.std() == actual.std(), "Embedding standard deviation mismatch"


def test_sparse_e5():
    # Arrange
    input_texts = [
        "query: how much protein should a female eat",
        "query: summit define",
        (
            "passage: As a general guideline, the CDC's average requirement of "
            "protein for women ages 19 to 70 is 46 grams per day. But, as you "
            "can see from this chart, you'll need to increase that if you're "
            "expecting or training for a marathon. Check out the chart below to "
            "see how much protein you should be eating each day."
        ),
        (
            "passage: Definition of summit for English Language Learners. : "
            "1  the highest point of a mountain : the top of a mountain. : "
            "2  the highest level. : 3  a meeting or series of meetings between "
            "the leaders of two or more governments."
        ),
    ]

    e5 = AutoModel.from_pretrained("intfloat/e5-small")

    # Act
    embs, _ = run_e5(e5, input_texts)
    one_hot_embs, _ = run_one_hot_e5(DenseE5(e5), input_texts)

    # Assert
    assert torch.allclose(embs, one_hot_embs, atol=1e-4), "Embeddings mismatch"
