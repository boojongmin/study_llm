from transformers import BertForMaskedLM, BertTokenizer, AdamW
import torch
from torch.utils.data import DataLoader, Dataset

# model = BertForMaskedLM.from_pretrained("model")
model_name = 'bert-base-uncased'
model = BertForMaskedLM.from_pretrained(model_name)

# print(model)

# BertForMaskedLM(
#   (bert): BertModel(
#     (embeddings): BertEmbeddings(
#       (word_embeddings): Embedding(30522, 768, padding_idx=0)
#       (position_embeddings): Embedding(512, 768)
#       (token_type_embeddings): Embedding(2, 768)
#       (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#       (dropout): Dropout(p=0.1, inplace=False)
#     )
#     (encoder): BertEncoder(
#       (layer): ModuleList(
#         (0-11): 12 x BertLayer(
#           (attention): BertAttention(
#             (self): BertSdpaSelfAttention(
#               (query): Linear(in_features=768, out_features=768, bias=True)
#               (key): Linear(in_features=768, out_features=768, bias=True)
#               (value): Linear(in_features=768, out_features=768, bias=True)
#               (dropout): Dropout(p=0.1, inplace=False)
#             )
#             (output): BertSelfOutput(
#               (dense): Linear(in_features=768, out_features=768, bias=True)
#               (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#               (dropout): Dropout(p=0.1, inplace=False)
#             )
#           )
#           (intermediate): BertIntermediate(
#             (dense): Linear(in_features=768, out_features=3072, bias=True)
#             (intermediate_act_fn): GELUActivation()
#           )
#           (output): BertOutput(
#             (dense): Linear(in_features=3072, out_features=768, bias=True)
#             (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#         )
#       )
#     )
#   )
#   (cls): BertOnlyMLMHead(
#     (predictions): BertLMPredictionHead(
#       (transform): BertPredictionHeadTransform(
#         (dense): Linear(in_features=768, out_features=768, bias=True)
#         (transform_act_fn): GELUActivation()
#         (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#       )
#       (decoder): Linear(in_features=768, out_features=30522, bias=True)
#     )
#   )
# )

# 예측 함수
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

tokenizer = BertTokenizer.from_pretrained(model_name)
text = "I am [MASK]."
predicted_token = predict_masked_token(model, tokenizer, text)
print(f"The predicted word is: {predicted_token}")