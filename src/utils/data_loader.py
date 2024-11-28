import pandas as pd
import pathlib


class DataLoader:
    def __init__(self):

        root = pathlib.Path(__file__).parent.parent.parent
        self.path = str(root) + "/data"

        self.classification_datasets = [
            "c_e_class_financial_distress.csv",
            "c_s_class_ny_arrests.csv",
        ]
        self.forcasting_datasets = [
            "f_e_forecast_gdp_europe.csv",
            "f_s_forecast_ny_arrests.csv",
        ]
        self.classification_targets = {
            "e": "CLASS",
            "s": "LAW_CAT_CD",
        }

        self.forecasting_targets = {
            "e": "GDP",
            "s": "Manhattan",
        }

    def get_econmical_classification_dataset_and_target(self):
        """
        Returns the economic classification dataset and target as a tuple.
        """
        data = pd.read_csv(self.path + "/" + self.classification_datasets[0])
        target = self.classification_targets["e"]
        return data, target

    def get_security_classification_dataset_and_target(self):
        """
        Returns the security classification dataset and target as a tuple.
        """
        data = pd.read_csv(self.path + "/" + self.classification_datasets[1])
        target = self.classification_targets["s"]
        return data, target

    def get_economic_forecasting_dataset_and_target(self):
        """
        Returns the economic forecasting dataset and target as a tuple.
        """
        data = pd.read_csv(self.path + "/" + self.forcasting_datasets[0])
        target = self.forecasting_targets["e"]
        return data, target

    def get_security_forecasting_dataset_and_target(self):
        """
        Returns the security forecasting dataset and target as a tuple.
        """
        data = pd.read_csv(self.path + "/" + self.forcasting_datasets[1])
        target = self.forecasting_targets["s"]
        return data, target
