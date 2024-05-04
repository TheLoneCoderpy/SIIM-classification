import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import numpy as np

def load_and_transform(file_name: str, export: bool = False):
    test = file_name.split(".")[0] == "test"

    gender_map = {"female": 0, "male": 1}
    df = pd.read_csv(file_name)

    to_drop = ["diagnosis", "benign_malignant"]

    if not export:
        df = df.dropna(axis=0)
    else:
        df = df.fillna(0)
    df = df.rename(columns={"sex": "gender"})
    df["gender"] = df["gender"].map(gender_map)
    #df["patient_id"] = df["patient_id"].str.split("_").str[1].astype(int)
    df = df.drop("patient_id", axis=1)
    df["age_approx"] = df["age_approx"].astype(int)
    
    anatom_i = 0
    anatom_site_map = {}
    for row in df.iterrows():
        if row[1]["anatom_site_general_challenge"] not in anatom_site_map:
            anatom_site_map[row[1]["anatom_site_general_challenge"]] = anatom_i
            anatom_i += 1
    
    if not test:
        df["target"] = df["target"].astype(float)
        

    df["anatom_site_general_challenge"] = df["anatom_site_general_challenge"].map(anatom_site_map)
        
    for v in to_drop:
        try:
            df = df.drop(columns=[v], axis=1)
        except:
            pass

    return df, anatom_site_map, gender_map


def fix_imbalance(df: pd.DataFrame): # Die Ungleichverteilung der Zielvariable in Dataset beheben
    ros = RandomOverSampler(sampling_strategy='all')
    smote = SMOTE(sampling_strategy='all')

    # Determine whether to use RandomOverSampler or SMOTE
    if df['target'].nunique() <= 2:
        # For binary columns, use RandomOverSampler
        df_resampled, _ = ros.fit_resample(df, df['target'])
    else:
        # For non-binary columns, use SMOTE
        # Ensure there are enough samples for SMOTE
        if len(df) >= smote.k_neighbors + 1:
            df_resampled, _ = smote.fit_resample(df, df['target'])
        else:
            # If not enough samples, use RandomOverSampler
            df_resampled, _ = ros.fit_resample(df, df['target'])

    return df_resampled