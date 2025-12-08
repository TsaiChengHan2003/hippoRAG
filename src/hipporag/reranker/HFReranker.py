import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class HFReranker:
    def __init__(self):
        # 1. 載入 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')
        
        # 2. 載入模型 (針對分類任務，num_labels=1 代表輸出一個分數)
        # 如果您有自己微調過的權重，就把 'microsoft/deberta-v3-base' 換成您的資料夾路徑
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'cross-encoder/ms-marco-MiniLM-L6-v2', 
            num_labels=1  # 回歸任務 (Regression) 或 二元分類的 Logits
        )
        self.model.eval() # 設定為評估模式

    def compute_score(self, query, doc):
        # 3. 準備輸入 (Cross-Encoder 格式)
        # Hugging Face 會自動處理 [CLS] Q [SEP] D [SEP] 的拼接
        inputs = self.tokenizer(
            query, 
            doc, 
            return_tensors='pt', # 回傳 PyTorch Tensor
            truncation=True,     # 過長截斷
            max_length=512,      # 最大長度
            padding=True         # 自動補 0
        )
        
        # 4. 推論 (Forward Pass)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # outputs.logits 形狀是 [batch_size, 1]
            score = outputs.logits.squeeze(-1).item()
            
        return score
    
    def compute_scores(self, query: str, docs: list) -> list:
        """
        [修正重點] 使用批次處理一次性計算所有文件的分數，大幅提升效能。
        """
        # 1. 準備批次輸入 (將 query 和所有 docs 配對)
        # 建立一個列表，格式為 [[query, doc1], [query, doc2], ...]
        texts = [[query, doc] for doc in docs]

        # 2. Tokenizer 批次處理 (一次性處理所有配對)
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
            max_length=512
        )
        
        # 3. 推論 (一次 Forward Pass)
        with torch.no_grad():
            # 將輸入資料移動到模型所在的設備（例如 GPU/CPU）
            # 假設 self.model.device 已定義
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()} 
            outputs = self.model(**inputs)
            
            # 4. 提取分數並轉換成 Python list
            # outputs.logits 的形狀為 [批次大小, 1]
            
            # --- 【 Sigmoid 歸一化核心步驟 】 ---
            
            # a. 應用 Sigmoid 函式
            # Sigmoid(x) = 1 / (1 + e^(-x))，將數值範圍壓縮到 [0, 1]
            sigmoid_scores = torch.sigmoid(outputs.logits)
            
            # b. 移動到 CPU，轉換為 NumPy 陣列，展平，再轉為 List
            scores = sigmoid_scores.cpu().numpy().flatten().tolist()
            
            formatted_scores = [f"{s:.6f}" for s in scores]

            # print(">>> [DEBUG] Normalized Sigmoid Scores:")
            # print(formatted_scores)

            # --- 【 步驟結束 】 ---
            
        return scores
