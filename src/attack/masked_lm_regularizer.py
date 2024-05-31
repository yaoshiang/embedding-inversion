"""Regularization of a token sequence.

This module contains the BertRegularizer class,
which regularizes the sparse token ids by penalizing low prob tokens per BERT.
"""

from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from transformers import AutoModelForMaskedLM, AutoTokenizer


# Check for device availability
def get_device() -> torch.device:
    """Returns the device to use for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")  # Prefer CUDA if available
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Use MPS if CUDA is not available and MPS is
    else:
        return torch.device("cpu")  # Default to CPU if neither CUDA nor MPS is available


# Set the device based on availability
DEVICE = get_device()
print("Using device:", DEVICE)


class MaskedLMRegularizer:
    """Regularizes the sparse token ids by penalizing low prob tokens per a maskedLM."""

    def __init__(
        self,
        device: torch.device,
        tokenizer=None,
        model: Optional[AutoModelForMaskedLM] = None,
        mask_prob: float = 0.15,
    ):
        """Initializes the BertRegularizer.

        Args:
            device (torch.device): The device to use for computation.
            tokenizer (Optional[AutoTokenizer]): The tokenizer to use.
            model (Optional[AutoModelForMaskedLM]): The model to use.
            mask_prob (Optional[float]): The probability of masking a token.
        """
        self.device = device
        self.mask_prob = mask_prob

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        if model is None:
            self.model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(self.device)

    def __call__(self, pred_tokens: torch.Tensor, log_probs: bool, generator:Optional[torch.Generator] = None) -> torch.Tensor:
        """Regularizes the sparse token ids by penalizing low prob tokens per BERT.

        Randomly masks token positions according to the mask_prob, then
        asks Bert to predict on that masked token. Then calculates KL divergence.

        Bug: The tokens that are randomly selected may be pad tokens.

        Args:
            pred_tokens (torch.Tensor): The tokens to regularize. These must be
                probabilities or log probabilities (depending on the log_probs arg).
                They must not be the token ids (e.g. cardinal values).
            log_probs (bool): Whether the predicted token ids are log probabilities.
            generator (Optional[torch.Generator]): The random number generator. Pass
                in a fresh generator with a manually set seed to enforce reproducibility
                for testing. 

        Returns:
            torch.Tensor: The regularization loss. Add it to your
            total loss and call backward on the total loss.
        """
        assert pred_tokens.dtype in (torch.float32, torch.float64), "Expected float tensor."
        assert pred_tokens.dim() == 3, f"Expected 3D tensor, got {pred_tokens.dim()}D tensor."
        if log_probs:
            assert torch.min(pred_tokens) < 0, "Expected log probabilities, didn't get negative values."
            # Can't check for values above 1.0, as it's unlikely with 30,000 tokens.
        else:
            assert torch.min(pred_tokens) >= 0, "Expected probabilities, got negative values."
            assert torch.max(pred_tokens) <= 1, "Expected probabilities, got values above 1."

        B, S, V = pred_tokens.shape  # noqa: N806

        assert V == self.tokenizer.vocab_size, f"Vocab size mismatch: {V} vs {self.tokenizer.vocab_size}"

        # Argmax the token ids. This is the "input" to BERT.
        pred_token_ids = pred_tokens.argmax(dim=-1).int().to(self.device)  # BS

        decoded = self.tokenizer.batch_decode(list(*pred_token_ids), skip_special_tokens=False)
        print("decoded pred_token_ids")
        print(decoded)

        # Pick some indexes to mask.
        # We will be gathering into a 2D view of the pred_token_ids of shape (B*S, V).
        num_mask_tokens = int(self.mask_prob * B * S) // 1
        if num_mask_tokens == 0:
            num_mask_tokens = 1


        num_mask_tokens=1
        assert num_mask_tokens > 0, "Expected at least one mask token."
        mask_idxs = torch.randperm(B * S, device=self.device, generator=generator).to(self.device)[:num_mask_tokens]
        pred_token_ids.view(B * S)[mask_idxs] = self.tokenizer.mask_token_id

        assert len(mask_idxs) == 1

        print("decoded pred_token_ids with mask")
        decoded = self.tokenizer.batch_decode(list(*pred_token_ids), skip_special_tokens=False)
        print(decoded)

        # Assert that we have enough mask tokens.
        # Note we are broadcasting the where from BSV to V.
        assert pred_token_ids.shape == (B, S)

        assert (
            num_mask_tokens == (pred_token_ids == self.tokenizer.mask_token_id).sum()
        ), f"Expected {num_mask_tokens} mask tokens. Got {(pred_token_ids == self.tokenizer.mask_token_id).sum()}."

        # Run the model with the masked tokens.
        outputs = self.model(pred_token_ids)
        output_logits = outputs.logits
        del outputs

        # We expect
        assert torch.min(output_logits) < 0, "Expected log probabilities, didn't get negative values."
        assert torch.max(output_logits) > 0, "Expected log probabilities, didn't get values above 1."

        decoded = self.tokenizer.batch_decode(list(*output_logits.argmax(dim=-1).int()), skip_special_tokens=False)
        print("decoded", decoded)

        # Gather the inputs and outputs for the masked tokens.
        bert_outputs = F.log_softmax(output_logits.view(B * S, V)[mask_idxs, :], dim=-1)
        matching_inputs = F.log_softmax(pred_tokens.view(B * S, V)[mask_idxs, :], dim=-1)

        # Calculate the KL divergence between the predicted and actual token distributions.
        # This is the regularization loss.
        assert torch.allclose(F.kl_div(bert_outputs, bert_outputs, log_target=True, reduction="batchmean"), torch.zeros(1, device=self.device))
        loss = F.kl_div(target=bert_outputs, input=matching_inputs, log_target=True, reduction="batchmean")
        print("kl div", loss, bert_outputs, matching_inputs)

        return loss

