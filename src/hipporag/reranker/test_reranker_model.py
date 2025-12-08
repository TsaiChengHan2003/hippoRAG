import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    
    def compute_scores(self, query, docs):
        scores = []
        for doc in docs:
            score = self.compute_score(query, doc)
            scores.append(score)
        return scores


# --- 測試 ---
if __name__ == "__main__":
    try:
        reranker = HFReranker()
        # 測試兩個例子
        s1 = reranker.compute_score("Where is Paris?", "Paris is the capital of France.")
        s2 = reranker.compute_score("Where is Paris?", "I like to eat apples.")
        
        print(f"相關文檔分數: {s1:.4f}")
        print(f"無關文檔分數: {s2:.4f}")
        
        if s1 > s2:
            print("✅ 測試成功：模型能區分相關性！")
        else:
            print("❌ 測試失敗：模型分不出來 (可能權重沒載入成功或訓練失敗)")
            
    except Exception as e:
        print(f"發生錯誤: {e}")
        print("請檢查 model_path 是否正確，以及資料夾內是否有 config.json")