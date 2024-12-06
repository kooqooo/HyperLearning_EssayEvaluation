import copy
from datetime import datetime
from typing import List, Dict
import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import KFold
import pandas as pd

class EssayDataset(Dataset):
    def __init__(self, prompts: List[str], essays: List[str], scores: Dict[str, List[float]], 
                 tokenizer, score_type: str, max_length=512, stride=128):
        self.prompts = prompts
        self.essays = essays
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.score_type = score_type
        self.special_tokens_count = 3  # [CLS], [SEP], [SEP]

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        essay = self.essays[idx]
        
        # # 디버깅을 위한 원본 길이 출력
        # print(f"Processing item {idx}")
        # print(f"Original prompt length: {len(prompt)}")
        # print(f"Original essay length: {len(essay)}")
        
        # 프롬프트는 전체 길이의 20%로 제한
        max_prompt_length = (self.max_length - self.special_tokens_count) // 5
        
        # 프롬프트 토큰화 - truncation 적용
        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            max_length=max_prompt_length,
            truncation=True
        )
        
        # print(f"Tokenized prompt length: {len(prompt_tokens)}")
        
        # 에세이에 사용할 수 있는 최대 길이 계산
        max_essay_chunk_length = self.max_length - len(prompt_tokens) - self.special_tokens_count
        
        # 에세이를 청크로 나누기 전에 전체 토큰화
        essay_tokens = self.tokenizer.encode(
            essay,
            add_special_tokens=False,
            truncation=True,
            max_length=max_essay_chunk_length
        )
        
        # print(f"Tokenized essay length: {len(essay_tokens)}")
        
        chunks_input_ids = []
        chunks_attention_mask = []
        
        # 단일 청크로 처리
        sequence = (
            [self.tokenizer.cls_token_id]
            + prompt_tokens
            + [self.tokenizer.sep_token_id]
            + essay_tokens
            + [self.tokenizer.sep_token_id]
        )
        
        # 최대 길이 제한
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length-1] + [self.tokenizer.sep_token_id]
        
        # print(f"Final sequence length: {len(sequence)}\n")
        
        # Attention mask 및 패딩
        attention_mask = [1] * len(sequence)
        padding_length = self.max_length - len(sequence)
        
        if padding_length > 0:
            sequence += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
        
        # 최종 검증
        assert len(sequence) <= self.max_length, \
            f"Sequence length {len(sequence)} exceeds max_length {self.max_length}"
        
        # 텐서 변환
        input_ids = torch.tensor([sequence])
        attention_mask = torch.tensor([attention_mask])
        score = torch.tensor([self.scores[self.score_type][idx]], dtype=torch.float)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': score,
            'chunk_count': torch.tensor(1)
        }

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

def collate_fn(batch):
    max_chunk_count = max(item['chunk_count'].item() for item in batch)
    batch_size = len(batch)
    
    # 첫 번째 항목에서 sequence length 가져오기
    seq_length = batch[0]['input_ids'].size(-1)
    
    # 배치 텐서 초기화
    input_ids = torch.zeros((batch_size, max_chunk_count, seq_length), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_chunk_count, seq_length), dtype=torch.long)
    labels = torch.zeros((batch_size, max_chunk_count), dtype=torch.float)
    chunk_counts = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        chunk_count = item['chunk_count'].item()
        input_ids[i, :chunk_count] = item['input_ids']
        attention_mask[i, :chunk_count] = item['attention_mask']
        labels[i, :chunk_count] = item['labels']
        chunk_counts[i] = chunk_count
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'chunk_counts': chunk_counts
    }

class EssayScorer:
    def __init__(
        self,
        model_name: str = "klue/roberta-base",
        use_cuda: bool = True,
        max_length: int = 512,
        stride: int = 128
    ):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.stride = stride
        self.models = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    def train(
        self,
        prompts: List[str],
        essays: List[str],
        scores: Dict[str, List[float]],
        use_kfold: bool = False,
        n_folds: int = 5,
        output_dir: str = "./output",
        batch_size: int = 8,
        num_epochs: int = 1,
        validation_split: float = 0.2
    ):
        score_types = ['con_score', 'org_score', 'exp_score']
        results = {}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for score_type in score_types:
            print(f"\nTraining model for {score_type}")
            
            if use_kfold:
                results[score_type] = self._train_with_kfold(
                    prompts, essays, scores, score_type, n_folds,
                    output_dir, batch_size, num_epochs
                )
            else:
                results[score_type] = self._train_simple(
                    prompts, essays, scores, score_type,
                    output_dir, batch_size, num_epochs, validation_split
                )

        return results

    def _train_simple(
        self,
        prompts: List[str],
        essays: List[str],
        scores: Dict[str, List[float]],
        score_type: str,
        output_dir: str,
        batch_size: int,
        num_epochs: int,
        validation_split: float
    ):
        dataset = EssayDataset(
            prompts, essays, scores, self.tokenizer,
            score_type, self.max_length, self.stride
        )

        train_size = int((1 - validation_split) * len(dataset))
        train_dataset, val_dataset = random_split(
            dataset, [train_size, len(dataset) - train_size]
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn
        )

        model = EssayModel(self.model_name).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        best_val_loss = float('inf')
        best_model = None
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = model(input_ids, attention_mask, labels)
                    val_loss += outputs['loss'].item()
            
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
        
        # Save the best model
        model_save_path = f"{output_dir}/{self.timestamp}.pth"
        torch.save(best_model.state_dict(), model_save_path)
        print(f"Best model for {score_type} saved at {model_save_path}")

        self.models[score_type] = best_model
        return {'eval_loss': best_val_loss}


    def _train_with_kfold(
        self,
        prompts: list[str],
        essays: list[str],
        scores: dict[str, list[float]],
        score_type: str,
        n_folds: int,
        output_dir: str,
        batch_size: int,
        num_epochs: int
    ):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results = []
        best_loss = float('inf')
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(essays)):
            print(f"Training fold {fold + 1}/{n_folds}")
            
            # Prepare datasets for current fold
            fold_train_prompts = [prompts[i] for i in train_idx]
            fold_val_prompts = [prompts[i] for i in val_idx]
            fold_train_essays = [essays[i] for i in train_idx]
            fold_val_essays = [essays[i] for i in val_idx]
            
            fold_train_scores = {
                key: [scores[i] for i in train_idx]
                for key, scores in scores.items()
            }
            fold_val_scores = {
                key: [scores[i] for i in val_idx]
                for key, scores in scores.items()
            }
            
            train_dataset = EssayDataset(
                fold_train_prompts, fold_train_essays, fold_train_scores,
                self.tokenizer, score_type, self.max_length, self.stride
            )
            val_dataset = EssayDataset(
                fold_val_prompts, fold_val_essays, fold_val_scores,
                self.tokenizer, score_type, self.max_length, self.stride
            )
            
            # Initialize model for this fold
            model = EssayModel(self.model_name).to(self.device)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"{output_dir}/{self.timestamp}/fold-{fold}",
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=100,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss"
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
            )
            
            # Train the model
            trainer.train()
            
            # Evaluate
            eval_results = trainer.evaluate()
            fold_results.append(eval_results)
            
            print(f"Fold {fold + 1} Results:")
            print(eval_results)
            
            # Store the best model
            if fold == 0 or eval_results['eval_loss'] < best_loss:
                self.models[score_type] = model
                best_loss = eval_results['eval_loss']
        
        return fold_results

    def predict(self, prompts: List[str], essays: List[str]) -> Dict[str, List[float]]:
        predictions = {}
        
        for score_type, model in self.models.items():
            model = model.to(self.device)
            model.eval()
            dataset = EssayDataset(
                prompts,
                essays,
                {score_type: [0.0] * len(essays)}, # 점수를 담을 공간 미리 할당
                self.tokenizer,
                score_type,
                self.max_length,
                self.stride
            )
            dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
            
            score_predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # 모델 출력이 직접 텐서로 나옴
                    outputs = model(input_ids, attention_mask)
                    # outputs는 이미 logits 텐서
                    prediction = outputs.squeeze().mean().item()
                    score_predictions.append(prediction)
            
            predictions[score_type] = score_predictions
        
        return predictions

# 사용 예시
if __name__ == "__main__":
    # 데이터 준비
    test_size = 100

    data = pd.read_csv("data/data.csv")[:test_size]
    prompts = data["main_subject"].tolist()
    essays = data["essay"].tolist()
    scores = {
        'con_score': data["con_score"].tolist(),
        'org_score': data["org_score"].tolist(),
        'exp_score': data["exp_score"].tolist()
    }

    # 모델 초기화
    scorer = EssayScorer()

    # 학습 (k-fold 사용)
    results_kfold = scorer.train(
        prompts=prompts,
        essays=essays,
        scores=scores,
        use_kfold=False,
        n_folds=2
    )
    
    # 예측
    new_prompts = [""]
    new_essays = [""]
    predictions = scorer.predict(new_prompts, new_essays)
    print(predictions)