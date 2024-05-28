from transformers import BertForMaskedLM, BertTokenizer, AdamW
import torch
from torch.utils.data import DataLoader, Dataset

# 사전 학습된 모델 및 토크나이저 로드
model_name = 'bert-base-uncased'
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 데이터셋 준비
class MaskedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, correct_tokens, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.correct_tokens = correct_tokens
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()

        # [MASK] 토큰의 위치를 찾고 정답 토큰(여기서는 '햄버거')으로 대체
        mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[0]
        if len(mask_token_index) > 0:
            correct_token_id = self.correct_tokens[idx]
            labels[mask_token_index] = correct_token_id
        
        return input_ids, attention_mask, labels

# 예제 데이터셋
texts = [
    "I am [MASK]."
]
correct_tokens = tokenizer.convert_tokens_to_ids(["hamburger"])

dataset = MaskedTextDataset(texts, tokenizer, correct_tokens)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 옵티마이저 설정
optimizer = AdamW(model.parameters(), lr=5e-5)

# 훈련 루프
model.train()
for epoch in range(200):
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()  # 이전 배치의 그래디언트를 초기화합니다.
        loss.backward()  # 현재 배치의 손실에 대한 그래디언트를 계산합니다.
        optimizer.step()  # 옵티마이저를 사용하여 모델의 파라미터를 업데이트합니다.

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

print("Training complete.")

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

# 예제 사용법
text = "I am [MASK]."
predicted_token = predict_masked_token(model, tokenizer, text)
print(f"The predicted word is: {predicted_token}")


model.save_pretrained("model")