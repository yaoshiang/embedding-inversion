import torch

from .trainable_token_sequence import TrainableTokenSequence  # Assuming the class is in mymodule.py


def test_initialization():
    """Test the initialization of trainable tokens."""
    batch_size = 2
    sequence_length = 10
    vocab_size = 30000
    model_stub = type(
        "ModelStub",
        (object,),
        {"config": type("Config", (object,), {"max_position_embeddings": sequence_length, "vocab_size": vocab_size})},
    )

    # Create instance
    trainable_tokens = TrainableTokenSequence(batch_size, sequence_length, vocab_size, 0.5)

    # Check logits initialization
    assert trainable_tokens.token_logits.shape[0] == batch_size
    assert trainable_tokens.token_logits.shape[1] <= sequence_length  # Layer may hard code some tokens.
    assert trainable_tokens.token_logits.shape[2] == vocab_size
    assert trainable_tokens.token_logits.requires_grad, "Logits must require gradients."


def test_forward_pass():
    """Test the forward pass outputs probabilities."""
    batch_size = 2
    sequence_length = 10
    vocab_size = 30000
    trainable_tokens = TrainableTokenSequence(batch_size, sequence_length, vocab_size, 0.5)

    # Forward pass. Returns log probabilities.
    output = trainable_tokens.forward()

    # Check output shape and type
    assert output.shape == (batch_size, sequence_length, vocab_size)
    sum_probs = torch.exp(output).sum(dim=-1)
    # print(sum_probs, sum_probs.shape, sum_probs.dtype)
    ones = torch.ones_like(sum_probs)
    # print(ones, ones.shape, ones.dtype)
    assert torch.allclose(
        sum_probs, ones, atol=1e-5
    ), f"Output probabilities should sum to 1 across vocab dimension: {output.sum(dim=-1)}"


def test_training_update():
    """Test if gradients are computed and weights updated."""
    batch_size = 2
    sequence_length = 10
    vocab_size = 30000
    trainable_tokens = TrainableTokenSequence(batch_size, sequence_length, vocab_size, 0.5)

    # Dummy target for loss computation (normally this would be more complex)
    target = torch.randn(batch_size, sequence_length, vocab_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD([trainable_tokens.token_logits], lr=0.1)

    # Forward pass
    output = trainable_tokens.forward()
    loss = criterion(output, target)

    # Backward pass and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check if logits were updated
    assert torch.sum(trainable_tokens.token_logits).item() != 0, "Logits should be updated by optimizer."
