# import torch

# from attack.trainable_token_sequence import PASSAGE, QUERY, TrainableTokenSequence


# def test_initialization():
#     """Test the initialization of trainable tokens."""

#     # Arrange
#     batch_size = 2
#     sequence_length = 10
#     vocab_size = 30000
#     lengths = [3, 4]
#     types = [QUERY, PASSAGE]

#     # Act
#     trainable_tokens = TrainableTokenSequence(batch_size, sequence_length, vocab_size, lengths, types, 1, 0.5)

#     # Assert
#     assert trainable_tokens.sequences.shape[0] == batch_size
#     assert trainable_tokens.sequences.shape[1] <= sequence_length  # Layer may hard code some tokens.
#     assert trainable_tokens.sequences.shape[2] == vocab_size
#     assert trainable_tokens.sequences.requires_grad, "Logits must require gradients."


# def test_forward_pass():
#     """Test the forward pass outputs probabilities."""

#     # Arrange
#     batch_size = 2
#     sequence_length = 10
#     vocab_size = 30000
#     lengths = [3, 4]
#     types = [QUERY, PASSAGE]
#     trainable_tokens = TrainableTokenSequence(batch_size, sequence_length, vocab_size, lengths, types, 1, 0.5)

#     # Act
#     output = trainable_tokens.forward()

#     # Assert
#     # Check output shape and type
#     assert output.shape == (batch_size, sequence_length, vocab_size)

#     sum_probs = torch.exp(output).sum(dim=-1)
#     ones = torch.ones_like(sum_probs)
#     assert torch.allclose(
#         sum_probs, ones, atol=1e-2
#     ), f"Output probabilities should sum to 1 across vocab dimension: {sum_probs}"


# def test_training_update():
#     """Test if gradients are computed and weights updated."""
#     # Arrange
#     batch_size = 2
#     sequence_length = 10
#     vocab_size = 30000
#     lengths = [3, 4]
#     depth = 1
#     types = [QUERY, PASSAGE]
#     trainable_tokens = TrainableTokenSequence(batch_size, sequence_length, vocab_size, lengths, types, depth, 0.5)

#     # Dummy target for loss computation (normally this would be more complex)
#     target = torch.randn(batch_size, sequence_length, vocab_size)
#     criterion = torch.nn.MSELoss()

#     # Act and Assert: check params. 
#     for param, length in zip(trainable_tokens.parameters(), lengths):
#         assert param.shape == (length, vocab_size, depth)

#     # Act 
#     # Forward pass
#     optimizer = torch.optim.SGD(trainable_tokens.parameters(), lr=0.1)
#     output = trainable_tokens.forward()
#     loss = criterion(output, target)

#     # Backward pass and optimizer step
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # Assert
#     # Check if logits were updated
#     assert torch.sum(trainable_tokens.sequences).item() != 0, "Logits should be updated by optimizer."
