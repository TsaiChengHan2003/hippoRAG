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
        
        # 3. 初始化設備並將模型移動到設備上
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

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
    
    def compute_scores(self, query: str, docs: list, batch_size: int = 32) -> list:
        """
        使用批次處理，避免記憶體溢出
        """
        all_scores = []
        
        # 分批處理文檔
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            texts = [[query, doc] for doc in batch_docs]
            
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            )
            
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()} 
                outputs = self.model(**inputs)
                sigmoid_scores = torch.sigmoid(outputs.logits)
                scores = sigmoid_scores.cpu().numpy().flatten().tolist()
                all_scores.extend(scores)
        
        return all_scores