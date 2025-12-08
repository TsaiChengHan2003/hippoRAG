------如果要開vllm模式使llama 3.1:8B性能加倍 得去hugginface下載加速版 但70B的不可能跑得動 流程如下-----
1. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct 填寫問卷等他批准
2. 下指令 huggingface-cil login 去拿你的access token
3. 批准通過後下此指令 =>
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192

幫我注意一下如果你開在port 8000 底下的llm_base_url http://{ip位址}:{port} port的部分也要改
-------------------------------------------------------------------------------------------------

# =============== for 政翰、星瑀 懶人包 ===============
1. ssh去162 進入對應之conda環境 下指令
pip install -r requirements.txt (第一次要幫我跑一下這個)
OLLAMA_HOST=0.0.0.0:11434 /datas/store162/chenghan/ollama/bin/ollama serve
/datas/store162/chenghan/ollama/bin/ollama run llama3.1
2. 在163 進入對應之conda環境 下指令
dataset="{這邊記得改}"
python main.py --dataset $dataset --llm_base_url http://140.122.184.162:11434 --llm_name llama3.1 --embedding_name nvidia/NV-Embed-v2
## ==================================================

## =============== for 俞華 懶人包 ===================
1. ssh去bigram 進入對應之conda環境 下指令 
pip -r requirements.txt (第一次要幫我跑一下這個)
ollama serve
run (llama3.1 or llama 3.3)

2. 在其他跑得動embedding model的下指令
dataset="{這邊記得改}"
python main.py --dataset $dataset --llm_base_url http://192.168.1.156:11434 --llm_name {這邊請依據你llama run的去開給值 ex: llama3.1、llama3.3} --embedding_name nvidia/NV-Embed-v2
## =================================================


# 目前已知有架設llama的機器
140.122.184.162:11434 有 llama3.1 8B
192.168.1.156:11434 bigram 好像全都有 (碩1生好像連不進去 防火牆有擋的樣子，我跟星瑀進不去)

# dataset
dataset=sample
dataset=test2Wiki
dataset=2wikimultihopqa
dataset=hotpotqa
dataset=musique


# Run OpenAI reproduce
python main.py --dataset $dataset \
              --llm_base_url https://api.openai.com/v1 \
              --llm_name gpt-4o-mini \
              --embedding_name nvidia/NV-Embed-v2 \
              --rerank_graph_mode true \
              --rerank_model_name ms-marco-MiniLM-L6-v2 \
              --rerank_threshold 0.5 \
              --openie_mode offline

# Run Labllama reproduce (在訓練之前 請先把llama在對應的ip開起來)
python main.py --dataset $dataset \
              --llm_base_url http://140.122.184.162:11434 \
              --llm_name llama3.3 \
              --rerank_graph_mode true \
              --embedding_name nvidia/NV-Embed-v2 \
              --rerank_graph_mode true \
              --rerank_model_name ms-marco-MiniLM-L6-v2 \
              --rerank_threshold 0.5

# Run ollama (下此行指令只能在有裝設llama環境的電腦 !!! 目前專題生162 跟bigram有 但bigram應該沒辦法用那邊不知道會不會掛載我的資料)
OLLAMA_HOST=0.0.0.0:11434 /datas/store162/chenghan/ollama/bin/ollama serve <!-- 提供服務 -->
/datas/store162/chenghan/ollama/bin/ollama run llama3.1 <!-- 選定model: llama3.1 or llama3.3 其他的得額外下載 -->

# HippoRAG 論文/git 網址
github https://github.com/OSU-NLP-Group/HippoRAG
HippoRAG 2-b31b1b"(https://arxiv.org/abs/2502.14802)
HippoRAG 1-b31b1b"(https://arxiv.org/abs/2405.14831)

# reproduce\dataset資料夾中
# 有_corpus：為offline Indexing的 建立 KG + Embeddings database用的檔案
# 沒有_corpus：的為提供 query 測試用資料（問題 + 正確答案 + 對應文件） 只是測試準確率使用!!


# 環境變數
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/datas/store162/chenghan/hf_home
export 


