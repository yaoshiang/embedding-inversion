import torch

# Load pre-trained model and tokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

# Example sentence with all tokens masked
input_text = "The [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Predict all masked tokens
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs.logits

print(predictions.shape, predictions)
print("argmax", torch.argmax(predictions, dim=-1))
predicted_tokens = tokenizer.decode(torch.argmax(predictions, dim=-1)[0])

# Decode the predictions
# predicted_tokens = [tokenizer.decode(torch.argmax(prediction).item()) for prediction in predictions[0]]

predicted_sentence = " ".join(predicted_tokens)
print(predicted_sentence)
