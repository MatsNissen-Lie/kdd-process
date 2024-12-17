from imblearn.over_sampling import SMOTE
import pathlib
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler

root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root))

from src.utils.data_loader import DataLoader
from src.utils.dslabs_functions import encode_cyclic_variables


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


from enum import Enum


class EvaluationEnum(Enum):
    ENCODING = 0
    MISSING_VALUES = 1
    OUTLIERS = 2
    SCALING = 3
    BALANCE = 4
    LAST = 100


class Pipeline:
    def __init__(self, evaluation: EvaluationEnum = None):
        self.data_loader = DataLoader()
        self.evaluation = evaluation if evaluation else EvaluationEnum.LAST

    def get_security_classification_dataset_and_target(
        self, sample_size: int = None, random_state: int = None
    ) -> tuple[pd.DataFrame, str]:

        # -----------------------
        #  Balance the data
        # -----------------------

        if self.evaluation.value >= EvaluationEnum.BALANCE.value:
            sample_size = 100000
        else:
            sample_size = 5100

        data, target = self.data_loader.get_security_classification_dataset_and_target(
            sample_size, random_state
        )

        # -----------------------
        # Drop Irrelevant Columns
        # -----------------------
        drop_columns = [
            "X_COORD_CD",  # duplicates information from Latitude/Longitude
            "Y_COORD_CD",  # same as above
            "PD_DESC",  # unstructured and less informative than OFNS_DESC
            "ARREST_KEY",  # unique identifier not needed for traning
        ]
        data = data.drop(columns=drop_columns)

        # -----------------------
        # Fix Known Data Issues
        # -----------------------
        # Fixing known typo in OFNS_DESC
        column = "OFNS_DESC"
        value = "OTHER STATE LAWS (NON PENAL LA"
        fix = "OTHER STATE LAWS (NON PENAL LAW)"

        if value in data[column].values:
            idx = data[data[column] == value].index[0]
            data.loc[idx, column] = fix

        # -----------------------
        # Binary Encodings
        # -----------------------
        # LAW_CAT_CD: map {'F':1, 'M':0}
        if "LAW_CAT_CD" in data.columns:
            data["LAW_CAT_CD"] = data["LAW_CAT_CD"].map({"F": 1, "M": 0})

        # PERP_SEX: map {'M':1, 'F':0}
        if "PERP_SEX" in data.columns:
            data["PERP_SEX"] = data["PERP_SEX"].map({"M": 1, "F": 0})

        # CLASS__security: map {'NY':1, 'nonNY':0}
        if "CLASS__security" in data.columns:
            data["CLASS__security"] = data["CLASS__security"].map({"NY": 1, "nonNY": 0})

        # -----------------------
        # Encode AGE_GROUP
        # -----------------------
        age_map = {"<18": 0, "18-24": 1, "25-44": 2, "45-64": 3, "65+": 4}
        if "AGE_GROUP" in data.columns:
            data["AGE_GROUP"] = data["AGE_GROUP"].map(age_map)

        # -----------------------
        # Encode PERP_RACE
        # -----------------------
        if "PERP_RACE" in data.columns:
            unique_races = data["PERP_RACE"].unique()
            race_encoder = {
                race: idx for idx, race in enumerate(unique_races) if race != "UNKNOWN"
            }
            race_encoder["UNKNOWN"] = np.nan
            data["PERP_RACE"] = data["PERP_RACE"].map(race_encoder)

        # Depending on how you want to handle NaN values:
        # data = data.dropna(subset=["PERP_RACE"])
        # or consider imputation. For now, we leave as is.

        # -----------------------
        # Encode Symbolic Variables
        # -----------------------
        # Instead of full dummification, we chose numeric encoding for OFNS_DESC, LAW_CODE, ARREST_BORO
        for col in ["OFNS_DESC", "LAW_CODE", "ARREST_BORO"]:
            if col in data.columns:
                encoder = {val: idx for idx, val in enumerate(data[col].unique())}
                data[col] = data[col].map(encoder)

        # -----------------------
        # Date Features & Cyclical Encoding
        # -----------------------
        # Extract year, month, day_of_week, weekend indicator, day_of_year
        if "ARREST_DATE" in data.columns:
            data["ARREST_DATE"] = pd.to_datetime(data["ARREST_DATE"])
            data["YEAR"] = data["ARREST_DATE"].dt.year
            data["MONTH"] = data["ARREST_DATE"].dt.month
            data["DAY_OF_WEEK"] = data["ARREST_DATE"].dt.dayofweek  # Monday=0, Sunday=6
            data["IS_WEEKEND"] = data["DAY_OF_WEEK"].isin([5, 6]).astype(int)
            data["DAY_OF_YEAR"] = data["ARREST_DATE"].dt.dayofyear

            # Cyclical encoding for DAY_OF_YEAR
            encode_cyclic_variables(data, ["DAY_OF_YEAR"])

            # Drop original date and DAY_OF_YEAR column
            data = data.drop(columns=["ARREST_DATE", "DAY_OF_YEAR"], errors="ignore")
        if self.evaluation.value == EvaluationEnum.ENCODING.value:
            return data, target
        # -----------------------
        # Final Dataset
        # -----------------------
        data = data.dropna()

        if self.evaluation.value < EvaluationEnum.MISSING_VALUES.value:
            return data, target
        # -----------------------
        #  Standardize the data
        # -----------------------
        target_data: Series = data.pop(target)

        transf: StandardScaler = StandardScaler(
            with_mean=True, with_std=True, copy=True
        ).fit(data)
        df_zscore = DataFrame(transf.transform(data), index=data.index, columns=data.columns)
        df_zscore[target] = target_data
        data = df_zscore
        if self.evaluation.value == EvaluationEnum.SCALING.value:
            return data, target

        # -----------------------
        #  Balance the data
        # -----------------------
        target_count: Series = data[target].value_counts()
        positive_class = target_count.idxmin()
        negative_class = target_count.idxmax()

        df_positives: Series = data[data[target] == positive_class]
        df_negatives: Series = data[data[target] == negative_class]

        df_neg_sample: DataFrame = DataFrame(df_negatives.sample(len(df_positives)))
        df_under: DataFrame = pd.concat([df_positives, df_neg_sample], axis=0)
        data = df_under
        if self.evaluation.value == EvaluationEnum.BALANCE.value:
            return data, target

        return data, target

    def get_econmical_classification_dataset_and_target(
        self,
    ) -> tuple[pd.DataFrame, str]:
        # Load the raw data and target
        data, target = (
            self.data_loader.get_econmical_classification_dataset_and_target()
        )
        correlated_variables = [
            "x7",
            "x13",
            "x33",
            "x34",
            "x38",
            "x48",
            "x49",
            "x50",
            "x52",
            "x53",
            "x62",
            "x75",
            "x76",
            "x77",
            "x79",
            "x81",
        ]
        data = data.drop(columns=correlated_variables)

        if self.evaluation.value < EvaluationEnum.SCALING.value:
            return data, target

        # -----------------------
        #  Standardize the data
        # -----------------------
        target_data: Series = data.pop(target)

        transf: StandardScaler = StandardScaler(
            with_mean=True, with_std=True, copy=True
        ).fit(data)
        df_zscore = DataFrame(transf.transform(data), index=data.index, columns=data.columns)
        df_zscore[target] = target_data
        data = df_zscore
        if self.evaluation.value == EvaluationEnum.SCALING.value:
            return data, target

        # -----------------------
        #  Balance the data
        # -----------------------
        RANDOM_STATE = 42

        smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
        y = data.pop(target).values
        X: np.ndarray = data.values
        smote_X, smote_y = smote.fit_resample(X, y)
        df_smote: DataFrame = pd.concat(
            [DataFrame(smote_X), DataFrame(smote_y)], axis=1
        )
        df_smote.columns = list(data.columns) + [target]
        data = df_smote

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
