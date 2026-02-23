# download_model.py
from modelscope import snapshot_download

# 将模型下载到当前目录下的 qwen_model 文件夹
model_dir = snapshot_download('qwen/Qwen2.5-1.5B-Instruct', cache_dir='./qwen_model')
print(f"模型已下载至: {model_dir}")