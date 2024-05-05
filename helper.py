import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import numpy as np

def load_and_transform(file_name: str, export: bool = False):
    to_drop = ["diagnosis", "benign_malignant"]
    gender_map = {"female": 0, "male": 1}
    test = file_name.split(".")[0] == "test"

    df = pd.read_csv(file_name)
    if not export: # wenn es sich um die Daten für den train_df handelt, sollen na-Werte (Zeilen) entfernt werden, bei test-Daten nur gefüllt
        df = df.dropna(axis=0)
    else:
        df = df.fillna(0)

    df = df.rename(columns={"sex": "gender"})
    df["gender"] = df["gender"].map(gender_map)
    df = df.drop("patient_id", axis=1)
    df["age_approx"] = df["age_approx"].astype(int)
    
    # label-dict bauen
    anatom_i = 0
    anatom_site_map = {}
    for row in df.iterrows():
        if row[1]["anatom_site_general_challenge"] not in anatom_site_map:
            anatom_site_map[row[1]["anatom_site_general_challenge"]] = anatom_i
            anatom_i += 1

    df["anatom_site_general_challenge"] = df["anatom_site_general_challenge"].map(anatom_site_map)
    
    if not test: # nur wenn die Spalte vorhanden ist
        df["target"] = df["target"].astype(float)

    for v in to_drop:
        try:
            df = df.drop(columns=[v], axis=1)
        except:
            pass

    return df, anatom_site_map, gender_map



def fix_imbalance(df: pd.DataFrame): # Die Ungleichverteilung der Zielvariable in Dataset beheben
    ros = RandomOverSampler(sampling_strategy='all') # dupliziert zufällig Beispiele aus der geringer vertretenden Klassen, strategy=all -> alle Spalten bekommen die selbe Anzahl an samples wie die mit den meisten
    smote = SMOTE(sampling_strategy='all') # erzeugt neue Datenpunkte (Zeilen) aus mehreren bestehenden, startegy=all siehe oben

    if df['target'].nunique() <= 2: # RandomOverSample braucht weniger Datenpunkte als SMOTE
        df_resampled, _ = ros.fit_resample(df, df['target'])
    else:
        # sicherstellen dass SMOTE genug Datenpunkte hat
        if len(df) >= smote.k_neighbors + 1:
            df_resampled, _ = smote.fit_resample(df, df['target'])
        else:
            df_resampled, _ = ros.fit_resample(df, df['target'])

    return df_resampled