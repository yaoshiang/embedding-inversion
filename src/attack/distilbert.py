import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")


text = "This is a test of [MASK] automobile."
print("text", text)

inputs = tokenizer(text, return_tensors="pt")
print("inputs", inputs)

mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
print('mask_token_index', mask_token_index)

logits = model(**inputs).logits
print("logits.shape", logits.shape)
print("logits", logits)

mask_token_logits = logits[0, mask_token_index, :]
print("mask_token_logits.shape", mask_token_logits.shape)
print("mask_token_logits", mask_token_logits)

topk_values, topk_ids = torch.topk(mask_token_logits, 5, dim=-1)
print("topk_ids", topk_ids)
topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids[0])
print('topk_tokens', topk_tokens)