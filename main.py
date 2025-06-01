import torch
import torchvision.transforms as transforms
from PIL import Image
import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import csv
from hydra.utils import get_original_cwd
import os
import time


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = torch.hub.load(
        repo_or_dir="miccunifi/QualiCLIP", source="github", model="QualiCLIP"
    )
    model.eval().to(device)

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    metadata_path = Path(cfg.metadata_path)
    base_dir = Path(get_original_cwd())

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    results = []
    for entry in metadata:
        rel_img_path = entry["img_path"]
        img_path = metadata_path.parent / rel_img_path

        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            score = model(img_tensor)

        results.append((str(rel_img_path), float(score.item())))

    # Write to CSV
    os.makedirs(os.path.join(base_dir, "results/individual"), exist_ok=True)
    output_csv_path = base_dir / "results/individual" / "quali_clip_metric.csv"
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_path", "quali_clip_score"])
        writer.writerows(results)


if __name__ == "__main__":
    base_dir = Path(get_original_cwd())
    start_time = time.time()
    main()
    end_time = time.time() - start_time
    with open(base_dir / "results/individual" / 'quali_clip.txt', 'w') as f:
        f.write(f"TIME: {end_time}")
