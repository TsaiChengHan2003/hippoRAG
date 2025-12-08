from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig
import json

dataset = "2wikimultihopqa"
save_dir = "outputs/gpt-4o-mini_nvidia_NV-Embed-v2"

# 載入 corpus 與 sample
corpus = json.load(open(f"reproduce/dataset/{dataset}_corpus.json"))
samples = json.load(open(f"reproduce/dataset/{dataset}.json"))
queries = [s["question"] for s in samples]
gold_answers = [s["answer"] for s in samples]

# 初始化 HippoRAG 並重用既有索引
config = BaseConfig(save_dir=save_dir, dataset=dataset)
hippo = HippoRAG(global_config=config)

# 這次只做 QA
qa_results = hippo.rag_qa(queries=queries, gold_answers=gold_answers)

# 將結果存成檔案
import json, os
os.makedirs(f"{save_dir}/results", exist_ok=True)
with open(f"{save_dir}/results/qa_results.json", "w", encoding="utf-8") as f:
    json.dump(qa_results, f, ensure_ascii=False, indent=2)
