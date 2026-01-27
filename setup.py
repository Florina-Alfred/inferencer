import os
import urllib.request
from tqdm import tqdm

YOLOX_MODELS = {
    "yolox_l": "https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_l.onnx",
    "yolox_m": "https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_m.onnx",
    "yolox_s": "https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_s.onnx",
    "yolox_tiny": "https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_tiny.onnx",
    "yolox_nano": "https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_nano.onnx",
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

for model_name, url in YOLOX_MODELS.items():
    out_path = f"{model_name}.onnx"
    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping...")
        continue
    print(f"Downloading {model_name} from {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=model_name) as t:
        urllib.request.urlretrieve(url, filename=out_path, reporthook=t.update_to)
    print(f"Download complete: {out_path}")

