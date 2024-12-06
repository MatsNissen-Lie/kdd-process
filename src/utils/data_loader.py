import numpy as np
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
        self.classification_targets = {
            "e": "CLASS",  # these can be changed as they are set by the teacher.
            "s": "CLASS__security",  # Updated to reflect the new target variable
        }

        self.forecasting_targets = {
            "e": "GDP",
            "s": "Manhattan",
        }
        self.default_dict = {
            "sample_size": 5100,
            "random_state": 42,
        }

    def binarize_jurisdiction_code(self, data):
        if "JURISDICTION_CODE" not in data.columns:
            raise KeyError("The dataset does not contain 'JURISDICTION_CODE' column.")
        target = self.classification_targets["s"]
        data[target] = data["JURISDICTION_CODE"].apply(
            lambda x: "NY" if x < 3 else "nonNY"
        )
        return data

    def get_security_classification_dataset_and_target(
        self, sample_size=None, random_state=None
    ):
        """
        Returns the security classification dataset and target as a tuple,
        with the target variable binarized and dataset reduced to a specified number of rows.

        Parameters:
        - sample_size (int): Number of rows to sample.
        - random_state (int): Seed for reproducibility.

        Returns:
        - Tuple[pd.DataFrame, str]: The processed DataFrame and target column name.
        """
        if sample_size is None:
            sample_size = self.default_dict["sample_size"]
        if random_state is None:
            random_state = self.default_dict["random_state"]
        # Load the entire dataset
        data = pd.read_csv(self.path + "/" + self.classification_datasets[1])

        # Binarize the 'JURISDICTION_CODE' into 'CLASS'

        # Sample the dataset if it exceeds the desired sample size #teacher says we only need 5000 to have a statistically meaningful dataset
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=random_state).reset_index(
                drop=True
            )
        data = self.binarize_jurisdiction_code(data)

        target = self.classification_targets["s"]
        return data, target

    def get_econmical_classification_dataset_and_target(self):
        """
        Returns the economic classification dataset and target as a tuple.
        """
        data = pd.read_csv(self.path + "/" + self.classification_datasets[0])
        target = self.classification_targets["e"]
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
