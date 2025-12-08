import pickle

# 加載 Pickle 文件
with open('hipporag/outputs/2wikimultihopqa/gpt-4o-mini_nvidia_NV-Embed-v2/graph.pickle', 'rb') as f:
    obj = pickle.load(f)

# 顯示對象的類型和內容
print(type(obj))
print(obj)
