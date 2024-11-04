import transformers

from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


inputs = tokenizer.encode_plus(
    "I love chess",
    "I love soccer and martial arts",
    add_special_tokens=True,
    max_length=20,
    padding="max_length",
    truncation=True,
    return_token_type_ids=True,
    return_attention_mask=True
)

print("Input IDs: ", inputs["input_ids"])
print("Attention Mask:", inputs["attention_mask"])
print("Token type ids:", inputs["token_type_ids"])