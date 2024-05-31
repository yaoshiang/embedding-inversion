import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small')

input_texts = [
    'query: how much protein should a female eat',
    'query: summit define',
    "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small')
model = AutoModel.from_pretrained('intfloat/e5-small')

# Tokenize the input texts
batch_dict = tokenizer(input_texts,
                       max_length=512,
                       padding=True,
                       truncation=True,
                       return_tensors='pt')

print('batch_dict keys', list(batch_dict.keys()))
print('batch_dict', batch_dict)
model = AutoModel.from_pretrained('intfloat/e5-small)

print(model)

named_children = model.named_children()
print('number of children:', len(list(model.named_children())))
for name, module in model.named_children():
    print(name, module, type(module))

print('config type:', type(model.config))
print('config:', model.config)

