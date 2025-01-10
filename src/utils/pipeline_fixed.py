from enum import Enum
from imblearn.over_sampling import SMOTE
import pathlib
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Adjust this import to your environment
root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root))

# Make sure these are valid imports in your environment
from src.utils.data_loader import DataLoader
from src.utils.dslabs_functions import encode_cyclic_variables


class DataLoaderInterface:
    def get_security_classification_dataset_and_target(
        self, sample_size: int = 5000, random_state: int = 42
    ) -> tuple[pd.DataFrame, str]:
        """
        DEFINITION for an interface-like method. 
        Your real DataLoader will have the actual data reading logic.
        """
        pass

    def get_econmical_classification_dataset_and_target(
        self,
    ) -> tuple[pd.DataFrame, str]:
        pass

    def get_economic_forecasting_dataset_and_target(self) -> tuple[pd.DataFrame, str]:
        pass

    def get_security_forecasting_dataset_and_target(self) -> tuple[pd.DataFrame, str]:
        pass


class EvaluationEnum(Enum):
    ENCODING = 0
    MISSING_VALUES = 1
    OUTLIERS = 2
    SCALING = 3
    BALANCE = 4
    FEATURE_SELECTION = 5
    LAST = 100


class Pipeline:
    def __init__(self, evaluation: EvaluationEnum = None):
        self.data_loader = DataLoader()
        self.evaluation = evaluation if evaluation else EvaluationEnum.LAST

    # ==========================================================================
    # REFACTORED METHOD #1: Security Classification with Train/Test
    # ==========================================================================
    def get_security_classification_train_test(
        self, sample_size: int = None, random_state: int = 42, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        

        # -----------------------
        # (A) Decide on sample size
        # -----------------------
        if self.evaluation.value >= EvaluationEnum.BALANCE.value:
            sample_size = 100000
        else:
            sample_size = 5100

        # -----------------------
        # (B) Load Data & Identify Target
        # -----------------------
        data, target = self.data_loader.get_security_classification_dataset_and_target(
            sample_size, random_state
        )

        # -----------------------
        # (C) Drop Irrelevant Columns
        # -----------------------
        drop_columns = [
            "X_COORD_CD",
            "Y_COORD_CD",
            "PD_DESC",
            "ARREST_KEY",
            "JURISDICTION_CODE"
        ]
        data = data.drop(columns=drop_columns, errors="ignore")
     
        # -----------------------
        # (D) Fix Known Data Issues (typos, etc.)
        # -----------------------
        column = "OFNS_DESC"
        value = "OTHER STATE LAWS (NON PENAL LA"
        fix = "OTHER STATE LAWS (NON PENAL LAW)"
        if column in data.columns and value in data[column].values:
            idx = data[data[column] == value].index[0]
            data.loc[idx, column] = fix

        # -----------------------
        # (E) Binary / Numeric Encodings
        # -----------------------
        # LAW_CAT_CD: map {'F': 1, 'M': 0}
        if "LAW_CAT_CD" in data.columns:
            data["LAW_CAT_CD"] = data["LAW_CAT_CD"].map({"F": 1, "M": 0})

        # PERP_SEX: map {'M':1, 'F':0}
        if "PERP_SEX" in data.columns:
            data["PERP_SEX"] = data["PERP_SEX"].map({"M": 1, "F": 0})

        # CLASS__security: {'NY':1, 'nonNY':0}
        if "CLASS__security" in data.columns:
            data["CLASS__security"] = data["CLASS__security"].map({"nonNY": 0, "NY": 1})

        # AGE_GROUP
        age_map = {"<18": 0, "18-24": 1, "25-44": 2, "45-64": 3, "65+": 4}
        if "AGE_GROUP" in data.columns:
            data["AGE_GROUP"] = data["AGE_GROUP"].map(age_map)

        # PERP_RACE
        if "PERP_RACE" in data.columns:
            unique_races = data["PERP_RACE"].unique()
            race_encoder = {
                race: idx for idx, race in enumerate(unique_races) if race != "UNKNOWN"
            }
            race_encoder["UNKNOWN"] = np.nan
            data["PERP_RACE"] = data["PERP_RACE"].map(race_encoder)
            # If needed: data = data.dropna(subset=["PERP_RACE"])

        # Symbolic -> numeric
        for col in ["OFNS_DESC", "LAW_CODE", "ARREST_BORO"]:
            if col in data.columns:
                encoder = {val: idx for idx, val in enumerate(data[col].unique())}
                data[col] = data[col].map(encoder)

        # -----------------------
        # (F) Date Features & Cyclical Encoding
        # -----------------------
        if "ARREST_DATE" in data.columns:
            data["ARREST_DATE"] = pd.to_datetime(data["ARREST_DATE"])
            data["YEAR"] = data["ARREST_DATE"].dt.year
            data["MONTH"] = data["ARREST_DATE"].dt.month
            data["DAY_OF_WEEK"] = data["ARREST_DATE"].dt.dayofweek
            data["IS_WEEKEND"] = data["DAY_OF_WEEK"].isin([5, 6]).astype(int)
            data["DAY_OF_YEAR"] = data["ARREST_DATE"].dt.dayofyear

            encode_cyclic_variables(data, ["DAY_OF_YEAR"])  # your custom function

            data = data.drop(columns=["ARREST_DATE", "DAY_OF_YEAR"], errors="ignore")

        # -----------------------
        # (G) Drop any remaining NaNs if desired
        # -----------------------
        data = data.dropna()
       
        labels = data[target].unique()
        
        # -----------------------
        # Separate target
        # -----------------------
        y = data.pop(target)

        # -----------------------
        # (H) Train/Test Split
        # -----------------------
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,  # helps preserve class distribution
        )

        # -----------------------
        # (I) SCALE: fit on train, transform both
        # -----------------------
        if self.evaluation.value >= EvaluationEnum.SCALING.value:
            scaler = StandardScaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns,
            )
            X_min = X_train.min()
            X_train = X_train - X_min

            X_test = pd.DataFrame(
                scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns,
            )
            X_test = X_test - X_min

        # -----------------------
        # (J) BALANCE: undersample or oversample on train only
        # -----------------------
        if self.evaluation.value >= EvaluationEnum.BALANCE.value:
            # Example: simple undersampling
            train_data = X_train.copy()
            train_data[target] = y_train

            target_counts = train_data[target].value_counts()
            positive_class = target_counts.idxmin()
            negative_class = target_counts.idxmax()

            df_positives = train_data[train_data[target] == positive_class]
            df_negatives = train_data[train_data[target] == negative_class]
            df_neg_sample = df_negatives.sample(len(df_positives), random_state=random_state)

            df_under = pd.concat([df_positives, df_neg_sample], axis=0)
            X_train = df_under.drop(columns=[target])
            y_train = df_under[target]

        # -----------------------
        # (K) FEATURE SELECTION
        # -----------------------
        if self.evaluation.value >= EvaluationEnum.FEATURE_SELECTION.value:
            vars2drop = [
                'PD_CD',
                'KY_CD', 
                'OFNS_DESC', 
                'LAW_CAT_CD', 
                'ARREST_BORO', 
                'AGE_GROUP', 
                'PERP_SEX', 
                'PERP_RACE', 
                'Latitude', 
                'Longitude', 
                'YEAR', 
                'MONTH', 
                'DAY_OF_WEEK', 
                'IS_WEEKEND', 
                'DAY_OF_YEAR_sin', 
                'DAY_OF_YEAR_cos'
            ]
            # Drop from BOTH sets
            existing_train_cols = [v for v in vars2drop if v in X_train.columns]
            existing_test_cols = [v for v in vars2drop if v in X_test.columns]

            X_train = X_train.drop(columns=existing_train_cols, errors='ignore')
            X_test = X_test.drop(columns=existing_test_cols, errors='ignore')

        return X_train, X_test, y_train, y_test, target, labels

    # ==========================================================================
    #2: Economical Classification with Train/Test
    # ==========================================================================
    def get_economical_classification_train_test(
        self,
        random_state: int = 42,
        test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
       
        # -----------------------
        # (A) Load Data
        # -----------------------
        data, target = self.data_loader.get_econmical_classification_dataset_and_target()

        # -----------------------
        # (B) Drop correlated variables "Finacial ditress" dominates because it is directly correlated to the class 
        # -----------------------
        correlated_variables = [
            "x7", "x13", "x33", "x34", "x38", "x48", "x49", "x50",
            "x52", "x53", "x62", "x75", "x76", "x77", "x79", "x81", "Financial Distress"
        ]
        data = data.drop(columns=correlated_variables, errors="ignore")
        # -----------------------
        # Separate target
        # -----------------------
        labels = data[target].unique()
     
        y = data.pop(target)
        # -----------------------
        # (C) Train/Test Split BEFORE scaling or balancing
        # -----------------------
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # -----------------------
        # (D) SCALING: fit on train, transform test
        # -----------------------
        if self.evaluation.value >= EvaluationEnum.SCALING.value:
            scaler = StandardScaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns
            )
            X_min = X_train.min()
            X_train = X_train - X_min

            X_test = pd.DataFrame(
                scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns
            )
            X_test = X_test-X_min

        # -----------------------
        # (E) BALANCE (e.g. SMOTE or undersampling) - on train only
        # -----------------------
        if self.evaluation.value >= EvaluationEnum.BALANCE.value:
            smote = SMOTE(sampling_strategy="minority", random_state=random_state)
            X_train_np, y_train_np = smote.fit_resample(X_train, y_train)
            # Back to DataFrame
            X_train = pd.DataFrame(X_train_np, columns=X_train.columns)
            y_train = pd.Series(y_train_np, name=target)

        # -----------------------
        # (F) Feature selection
        # -----------------------
        if self.evaluation.value >= EvaluationEnum.FEATURE_SELECTION.value:
            vars2drop = [
                'x64', 'x12', 'x35', 'x27', 'x46', 'x17', 'x43', 'x25', 'x22', 'x18',
                'x54', 'x68', 'x1', 'x20', 'x21', 'x15', 'x57', 'x31', 'x44', 'x39'
            ]
            existing_train_cols = [v for v in vars2drop if v in X_train.columns]
            existing_test_cols = [v for v in vars2drop if v in X_test.columns]

            X_train = X_train.drop(columns=existing_train_cols, errors='ignore')
            X_test = X_test.drop(columns=existing_test_cols, errors='ignore')

        return X_train, X_test, y_train, y_test, target, labels


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # Example usage: Security Classification
    pipeline = Pipeline(evaluation=EvaluationEnum.LAST)
    X_train, X_test, y_train, y_test, target, labels = pipeline.get_security_classification_train_test()

    print("Security Classification Shapes:")
    print("  X_train:", X_train.shape, " y_train:", y_train.shape)
    print("  X_test :", X_test.shape,  " y_test :", y_test.shape)

    # Example usage: Economical Classification
    X_train_e, X_test_e, y_train_e, y_test_e, target, labels = pipeline.get_economical_classification_train_test()
    print("\nEconomical Classification Shapes:")
    print("  X_train:", X_train_e.shape, " y_train:", y_train_e.shape)
    print("  X_test :", X_test_e.shape,  " y_test :", y_test_e.shape)
