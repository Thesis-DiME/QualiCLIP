import os
from pathlib import Path
from typing import List, Dict
import json
import csv
from tqdm import tqdm
import torch
import hydra
from omegaconf import DictConfig
# from submodules.QualiClip.metric import 

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
    
class QualiCLIPPipeline:
    def __init__(self, cfg: DictConfig):
        """
        Initialize the QualiCLIPPipeline with a configuration object.

        Args:
            cfg (DictConfig): Configuration object containing pipeline settings.
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Load the QualiCLIPMetric instance
        self.metric = hydra.utils.instantiate(
            cfg.evaluation.qualiclip.metric,
            model_repo=cfg.evaluation.qualiclip.model_repo,
        ).to(self.device)

        # Load image data
        self.data = load_data(cfg.evaluation.qualiclip.data_file, "json")

    def evaluate(self):
        """
        Evaluate the images using the QualiCLIPMetric and save results.
        """
        print("Evaluating images...")
        for item in tqdm(self.data, desc="Processing Images", unit="image"):
            self.metric.update([item["img_path"]])

        # Compute results
        result_list = self.metric.compute()

        # Save results
        self._save_results(result_list)
        self._save_csv_results(result_list)

    def _save_csv_results(self, result_list: List[Dict]):
        """
        Save the computed results to a CSV file.

        Args:
            result_list (List[Dict]): List of results to save.
        """
        if not result_list:
            return

        # Flatten results for CSV compatibility
        flattened_results = []
        for result in result_list:
            flat_result = result.copy()
            for key, value in result.items():
                if isinstance(value, dict):
                    flat_result[key] = json.dumps(value, ensure_ascii=False)
            flattened_results.append(flat_result)

        # Handle existing CSV file
        file_exists = Path(self.cfg.csv_path).exists()
        existing_data = []

        if file_exists:
            with open(self.cfg.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)

            if len(existing_data) != len(flattened_results):
                raise ValueError(
                    f"Cannot append columns: Existing file has {len(existing_data)} rows "
                    f"but new data has {len(flattened_results)} rows"
                )

            for existing_row, new_row in zip(existing_data, flattened_results):
                existing_row.update(new_row)

            fieldnames = list(existing_data[0].keys()) if existing_data else []
            new_columns = [
                col for col in flattened_results[0].keys() if col not in fieldnames
            ]
            fieldnames.extend(new_columns)
        else:
            existing_data = flattened_results
            fieldnames = (
                list(flattened_results[0].keys()) if flattened_results else []
            )

        # Write updated data to CSV
        with open(self.cfg.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)

        print(f"Results appended to {self.cfg.csv_path}")

    def _save_results(self, result_list):
        """
        Save the computed results to a JSON file.

        Args:
            result_list (List[Dict]): List of results to save.
        """
        with open(self.cfg.save_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)

        print(f"Results saved to {self.cfg.save_path}")