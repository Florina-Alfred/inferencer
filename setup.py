import os
import urllib.request
from tqdm import tqdm
import argparse

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


def main():
    parser = argparse.ArgumentParser(description="Download YOLOX ONNX models.")
    parser.add_argument("-m", "--model", type=str, help="Download only a specific model: l, m, s, tiny, or nano")
    args = parser.parse_args()

    os.makedirs("model", exist_ok=True)
    short_to_name = {
        "l": "yolox_l",
        "m": "yolox_m",
        "s": "yolox_s",
        "tiny": "yolox_tiny",
        "nano": "yolox_nano"
    }

    models_to_download = YOLOX_MODELS.keys()
    if args.model:
        model_key = short_to_name.get(args.model)
        if model_key is None:
            print(f"Unknown model short name: {args.model}")
            print(f"Allowed: {', '.join(short_to_name.keys())}")
            exit(1)
        models_to_download = [model_key]

    for model_name in models_to_download:
        url = YOLOX_MODELS[model_name]
        out_path = os.path.join("model", f"{model_name}.onnx")
        if os.path.exists(out_path):
            print(f"{out_path} already exists, skipping...")
            continue
        print(f"Downloading {model_name} from {url}")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=model_name) as t:
            urllib.request.urlretrieve(url, filename=out_path, reporthook=t.update_to)
        print(f"Download complete: {out_path}")

if __name__ == "__main__":
    main()

