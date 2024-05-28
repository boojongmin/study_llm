from transformers import BertForMaskedLM, BertTokenizer
import torch

model_name = 'bert-base-uncased'
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def predict_masked_token(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs.input_ids

    masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits

    predicted_token_id = torch.argmax(predictions[0, masked_index], dim=-1)
    predicted_token = tokenizer.decode(predicted_token_id)

    return predicted_token

# 예제 사용법
text = "The capital of France is [MASK]."
tokenized_text = tokenizer.encode_plus(text, return_tensors='pt')
input_ids = tokenized_text['input_ids']
input_ids[0, input_ids[0] == tokenizer.mask_token_id] = tokenizer.mask_token_id

predicted_token = predict_masked_token(model, tokenizer, text)
print(f"The predicted word is: {predicted_token}")
