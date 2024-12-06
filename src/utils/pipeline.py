import pathlib
import sys

from numpy import nan


root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root))

from src.utils.data_loader import DataLoader
import pandas as pd
from pandas import DataFrame


class DataLoaderInterface:
    def get_security_classification_dataset_and_target(
        self, sample_size: int = 5000, random_state: int = 42
    ) -> tuple[pd.DataFrame, str]:
        pass

    def get_econmical_classification_dataset_and_target(
        self,
    ) -> tuple[pd.DataFrame, str]:
        pass

    def get_economic_forecasting_dataset_and_target(self) -> tuple[pd.DataFrame, str]:
        pass

    def get_security_forecasting_dataset_and_target(self) -> tuple[pd.DataFrame, str]:
        pass


class Pipeline:
    def __init__(self):
        self.data_loader = DataLoader()

    def get_security_classification_dataset_and_target(
        self, sample_size: int = None, random_state: int = None
    ) -> tuple[pd.DataFrame, str]:

        data, target = self.data_loader.get_security_classification_dataset_and_target(
            sample_size, random_state
        )
        drop_columns = [
            "X_COORD_CD",  # have 1 to 1 correlation to Latitude and Longitude
            "Y_COORD_CD",  # same as above
            "PD_DESC",  # unstructured and not useful. Just use the OFNS_DESC instead
        ]
        data = data.drop(columns=drop_columns)

        # fix one wierd row. where the two last characters are missing.
        indecies = data[data["OFNS_DESC"] == "OTHER STATE LAWS (NON PENAL LA"].index
        if len(indecies) > 0:
            index = indecies[0]
            data.loc[index, "OFNS_DESC"] = "OTHER STATE LAWS (NON PENAL LAW)"

        # we need to encode the data.
        age_map = {"18-24": 1, "25-44": 2, "45-64": 3, "65+": 4, "<18": 0}
        data["AGE_GROUP"] = data["AGE_GROUP"].map(age_map).fillna(nan)

        # # Encode race
        race_encoder = {
            race: idx for idx, race in enumerate(data["PERP_RACE"].unique())
        }
        # the encode has UNKNOWN as a key, should we remove it maybe?

        print(data["PERP_RACE"].unique())
        data["PERP_RACE"] = data["PERP_RACE"].map(race_encoder).fillna(nan)

        return data, target


if __name__ == "__main__":
    pipeline = Pipeline()
    data = pipeline.get_security_classification_dataset_and_target()

    data_with_nan = data[0].copy()
    # remove these cols PD_DESC  KY_CD OFNS_DESC
    data_with_nan = data_with_nan.drop(columns=["PD_DESC", "KY_CD", "OFNS_DESC"])

    # nan find nan values
    # using pd.isna()
    only_rows_with_nan = data_with_nan[
        pd.isna(data_with_nan["AGE_GROUP"]) | pd.isna(data_with_nan["PERP_RACE"])
    ]

    print(only_rows_with_nan)
