import pathlib
import sys

root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root))

from src.utils.data_loader import DataLoader
import pandas as pd


class Pipeline:
    def __init__(self):
        self.data_loader = DataLoader()
        datasets: dict[str, dict[str, tuple[pd.DataFrame, str]]] = {
            "classification": {
                "economic": self.data_loader.get_econmical_classification_dataset_and_target(),
                "security": self.data_loader.get_security_classification_dataset_and_target(),
            },
            "forecasting": {
                "economic": self.data_loader.get_economic_forecasting_dataset_and_target(),
                "security": self.data_loader.get_security_forecasting_dataset_and_target(),
            },
        }
