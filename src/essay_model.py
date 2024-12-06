import torch
from transformers import AutoModelForSequenceClassification


class EssayModel(torch.nn.Module):
    def __init__(self, model_name="klue/roberta-base"):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        # 입력이 3차원(batch_size, chunk_count, seq_length)인 경우 처리
        if len(input_ids.shape) == 3:
            batch_size, chunk_count, seq_length = input_ids.shape
            
            # 청크를 batch dimension으로 펼치기
            input_ids = input_ids.view(-1, seq_length)
            attention_mask = attention_mask.view(-1, seq_length)
            
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 로짓을 다시 청크별로 그룹화
            logits = logits.view(batch_size, chunk_count, -1)
            
            # 청크별 예측의 평균 계산
            logits = logits.mean(dim=1)
            
            loss = None
            if labels is not None:
                loss_fct = torch.nn.MSELoss()
                # chunk_count로 나누어진 labels 사용
                labels = labels.view(batch_size, chunk_count)[:, 0]
                loss = loss_fct(logits.squeeze(-1), labels)
        else:
            # 일반적인 2차원 입력 처리
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = None
            if labels is not None:
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.squeeze(-1), labels)
            
        return {'loss': loss, 'logits': logits} if loss is not None else logits