import torch
import torchvision.transforms as transforms
from PIL import Image
import torchmetrics
import csv
import os
import json


def load_csv_as_dict_list(file_path):
    dict_list = []
    with open(file_path, mode="r", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            dict_list.append(row)
    return dict_list


def load_json_as_dict_list(file_path):
    with open(file_path, mode="r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return data


def load_data(file_path, file_type):
    if file_type == "csv":
        return load_csv_as_dict_list(file_path)
    elif file_type == "json":
        return load_json_as_dict_list(file_path)
    else:
        raise ValueError("Unsupported file type. Please use 'csv' or 'json'.")
    
class QualiCLIPMetric(torchmetrics.Metric):
    def __init__(self, model_repo="miccunifi/QualiCLIP"):
        """
        Custom metric class for computing image quality scores using QualiCLIP.

        Args:
            model_repo (str): Repository path for loading the QualiCLIP model.
            device (torch.device): Device to run the model on (default: auto-detect CUDA if available).
        """
        super().__init__()

        # Load the model
        self.model = torch.hub.load(
            repo_or_dir=model_repo, source="github", model="QualiCLIP"
        )
        self.model.eval().to(self.device)

        # Define the preprocessing pipeline
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        # Add state variables
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, images):
        """
        Update the metric state with a batch of images.

        Args:
            images (List[str] or List[PIL.Image.Image]): List of image file paths or PIL Images.
        """
        # Ensure images are in list format
        if not isinstance(images, list):
            images = [images]

        # Process each image and compute its score
        with torch.no_grad(), torch.cuda.amp.autocast():
            for img in images:
                # Load and preprocess the image if it's a file path
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")

                # Preprocess the image
                img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

                # Compute the quality score
                score = self.model(img_tensor)

                # Store the score
                self.scores.append(score.item())

    def compute(self):
        """
        Compute the final metric value based on the accumulated state.

        Returns:
            float: Average quality score across all images.
        """
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def write_results(self, output_file, image_paths):
        """
        Write the computed scores for each image to a file.

        Args:
            output_file (str): Path to the output file (CSV or JSON).
            image_paths (List[str]): List of image file paths corresponding to the scores.
        """
        if len(self.scores) != len(image_paths):
            raise ValueError(
                "The number of scores does not match the number of image paths."
            )

        # Determine the file format based on the extension
        ext = os.path.splitext(output_file)[1].lower()
        if ext == ".csv":
            self._write_csv(output_file, image_paths)
        elif ext == ".json":
            self._write_json(output_file, image_paths)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json.")

    def _write_csv(self, output_file, image_paths):
        """
        Write the results to a CSV file.

        Args:
            output_file (str): Path to the output CSV file.
            image_paths (List[str]): List of image file paths.
        """
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Path", "Quality Score"])
            for img_path, score in zip(image_paths, self.scores):
                writer.writerow([img_path, score])

    def _write_json(self, output_file, image_paths):
        """
        Write the results to a JSON file.

        Args:
            output_file (str): Path to the output JSON file.
            image_paths (List[str]): List of image file paths.
        """
        import json

        results = [
            {"image_path": img_path, "quality_score": score}
            for img_path, score in zip(image_paths, self.scores)
        ]
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)


# Example usage
if __name__ == "__main__":
    metric = QualiCLIPMetric()

    img_paths = [
        "/home/naumov/code/general-pipeline/data/generated_images/stable-diffusion-v1-5-stable-diffusion-v1-5/3/image_0.png",
        "/home/naumov/code/general-pipeline/data/generated_images/stable-diffusion-v1-5-stable-diffusion-v1-5/3/image_1.png",
        "/home/naumov/code/general-pipeline/data/generated_images/stable-diffusion-v1-5-stable-diffusion-v1-5/3/image_2.png",
        "/home/naumov/code/general-pipeline/data/generated_images/stable-diffusion-v1-5-stable-diffusion-v1-5/3/image_3.png",
        "/home/naumov/code/general-pipeline/data/generated_images/stable-diffusion-v1-5-stable-diffusion-v1-5/3/image_4.png",
        "/home/naumov/code/general-pipeline/data/generated_images/stable-diffusion-v1-5-stable-diffusion-v1-5/3/image_5.png",
        "/home/naumov/code/general-pipeline/data/generated_images/stable-diffusion-v1-5-stable-diffusion-v1-5/3/image_6.png",
        "/home/naumov/code/general-pipeline/data/generated_images/stable-diffusion-v1-5-stable-diffusion-v1-5/3/image_7.png",
    ]

    metric.update(img_paths)

    avg_score = metric.compute()
    print(f"Average image quality score: {avg_score}")

    output_file = "image_quality_scores.csv"
    metric.write_results(output_file, img_paths)
    print(f"Results written to {output_file}")
