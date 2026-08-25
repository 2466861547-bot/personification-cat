"""下载 CosyVoice2 模型并查看结构"""
import os
from modelscope import snapshot_download

model_dir = snapshot_download('iic/CosyVoice2-0.5B')
print(f'Model dir: {model_dir}')

for root, dirs, files in os.walk(model_dir):
    for f in files:
        fpath = os.path.join(root, f)
        size = os.path.getsize(fpath)
        relpath = os.path.relpath(fpath, model_dir)
        print(f'  {relpath} ({size/1024/1024:.1f}MB)')