import torch
from torch.utils.data import Dataset


class EssayDataset(Dataset):
    def __init__(self, prompts: list[str], essays: list[str], scores: dict[str, list[float]], 
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

        # 프롬프트는 전체 길이의 20%로 제한
        max_prompt_length = (self.max_length - self.special_tokens_count) // 5

        # 프롬프트 토큰화 - truncation 적용
        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            max_length=max_prompt_length,
            truncation=True
        )

        # 에세이에 사용할 수 있는 최대 길이 계산
        max_essay_chunk_length = self.max_length - len(prompt_tokens) - self.special_tokens_count
        
        # 에세이를 청크로 나누기 전에 전체 토큰화
        essay_tokens = self.tokenizer.encode(
            essay,
            add_special_tokens=False,
            truncation=True,
            max_length=max_essay_chunk_length
        )

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
        
        # Attention mask 및 패딩
        attention_mask = [1] * len(sequence)
        padding_length = self.max_length - len(sequence)
        
        if padding_length > 0:
            sequence += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
        
        # 테스트
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