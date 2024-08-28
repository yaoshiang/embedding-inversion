import torch
import torch.nn.functional as F  # noqa: N812

from attack import masked_lm_regularizer


def test_default_args():
    # Act
    reg = masked_lm_regularizer.MaskedLMRegularizer(device=masked_lm_regularizer.DEVICE)

    # Assert
    assert reg.tokenizer is not None
    assert reg.model is not None


def phrase_to_logprobs(phrase: str, reg: masked_lm_regularizer.MaskedLMRegularizer):
    ids = reg.tokenizer([phrase], return_tensors="pt").to(masked_lm_regularizer.DEVICE)
    hots = F.one_hot(ids["input_ids"], num_classes=reg.tokenizer.vocab_size).float()
    clipped = torch.clamp(hots, 0.000001, 0.999999)
    logits = torch.log(clipped)

    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs


def test_unlikely_phrase():
    # Arrange
    reg = masked_lm_regularizer.MaskedLMRegularizer(device=masked_lm_regularizer.DEVICE)

    test_cases = [
        ("I love you.", "I love r."),
        ("I love you.", "you. love I"),
        ("The third baseman caught the ball.", "The third baseman caught the pow."),
        ("The third baseman caught the ball.", "caught ball the. The third baseman"),
        ("A cat is a mammal.", "a cat is a +."),
        ("A cat is a mammal.", "cat mammal is a."),
        ("You can see.", "You see minus."),
        ("You can see.", "see can You."),
    ]

    for normal, unusual in test_cases:
        for loss_type in ["kl_div", "smooth_l1"]:
            normal_logprobs = phrase_to_logprobs(normal, reg)
            normal_loss = reg(
                normal_logprobs,
                loss_type=loss_type,
            )
            unusual_logprobs = phrase_to_logprobs(unusual, reg)
            unusual_loss = reg(
                unusual_logprobs,
                loss_type=loss_type,
            )
            assert (
                unusual_loss > normal_loss
            ), f"Expected {unusual} {unusual_loss} to have higher loss than {normal} {normal_loss}."
