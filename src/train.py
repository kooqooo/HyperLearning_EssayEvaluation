import copy
from datetime import datetime
from typing import List, Dict
import os

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import KFold
import pandas as pd

from essay_dataset import EssayDataset
from essay_model import EssayModel

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
            
            # fold에 해당하는 데이터셋 준비
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
            
            # 각 fold에 해당하는 데이터로 모델 학습
            model = EssayModel(self.model_name).to(self.device)
            
            # 하이퍼파라미터 설정
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
            
            # Trainer 초기화
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
            )
            
            # 학습
            trainer.train()
            
            # 평가
            eval_results = trainer.evaluate()
            fold_results.append(eval_results)
            
            print(f"Fold {fold + 1} Results:")
            print(eval_results)
            
            # 모델 저장
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
        use_kfold=True,
        n_folds=5
    )
    
    # 예측 (LLM으로 생성한 데이터 사용)
    new_prompts = [""]
    new_essays = [""]
    predictions = scorer.predict(new_prompts, new_essays)
    print(predictions)