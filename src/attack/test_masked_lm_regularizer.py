import torch.nn.functional as F  # noqa: I001

from . import masked_lm_regularizer

import torch


def test_default_args():
    # Act
    reg = masked_lm_regularizer.MaskedLMRegularizer(device=masked_lm_regularizer.DEVICE)

    # Assert
    assert reg.mask_prob == 0.15
    assert reg.tokenizer is not None
    assert reg.model is not None


def phrase_to_logprobs(phrase: str, reg: masked_lm_regularizer.MaskedLMRegularizer):
    ids = reg.tokenizer([phrase], return_tensors="pt").to(masked_lm_regularizer.DEVICE)
    hots = F.one_hot(ids["input_ids"], num_classes=reg.tokenizer.vocab_size).float()
    clipped = torch.clamp(hots, 0.0001, 0.9999)
    logits = torch.log(clipped)

    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs


def test_unlikely_phrase():
    # Arrange
    reg = masked_lm_regularizer.MaskedLMRegularizer(device=masked_lm_regularizer.DEVICE)

    common_phrases = [
        "I am a person",
    ]
    weird_phrases = [
        "fielder fluttering + space",
    ]

    for common, weird in zip(common_phrases, weird_phrases):
        common_logprobs = phrase_to_logprobs(common, reg)
        common_loss = reg(
            common_logprobs,
            log_probs=True,
            generator=torch.Generator(device=masked_lm_regularizer.DEVICE).manual_seed(42),
        )
        weird_logprobs = phrase_to_logprobs(weird, reg)
        weird_loss = reg(
            weird_logprobs,
            log_probs=True,
            generator=torch.Generator(device=masked_lm_regularizer.DEVICE).manual_seed(42),
        )

        assert (
            weird_loss > common_loss
        ), f"Expected {weird} {weird_loss} to have higher loss than {common} {common_loss}."
