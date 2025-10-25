import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


PATIENT_ATTRS = {
    "sigla_uf_residencia", "idade_paciente", "sexo_paciente", 
    "raca_cor_paciente", "gestante_paciente",
    "dias_sintomas_notificacao",

    "possui_doenca_autoimune", "possui_diabetes", "possui_doencas_hematologicas",
    "possui_hepatopatias", "possui_doenca_renal", "possui_hipertensao",
    "possui_doenca_acido_peptica",

    "apresenta_febre", "apresenta_cefaleia", "apresenta_exantema",
    "apresenta_dor_costas", "apresenta_mialgia", "apresenta_vomito", 
    "apresenta_conjutivite", "apresenta_dor_retroorbital", "apresenta_artralgia", 
    "apresenta_artrite", "apresenta_leucopenia", "apresenta_petequias", 
    "prova_laco",
}

BINARY_ATTRS = {
    key for key in PATIENT_ATTRS if key.startswith(("possui_", "apresenta_"))
}
NUMERIC_ATTRS = {"idade_paciente", "dias_sintomas_notificacao"}
CATEGORICAL_ATTRS = PATIENT_ATTRS - BINARY_ATTRS - NUMERIC_ATTRS


def uf_to_region(uf: str) -> str:
    """Converts UF (attribute "sigla_uf_residencia") into region."""
    if uf is None:
        return None 
    
    uf = uf.upper()
    if uf in {"AC", "AP", "AM", "PA", "RO", "RR", "TO"}:
        return "N" # North
    if uf in {"AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"}:
        return "NE" # Northeast
    if uf in {"DF", "GO", "MT", "MS"}:
        return "CO" # Central-West
    if uf in {"ES", "MG", "RJ", "SP"}:
        return "SE" # Southeast
    if uf in {"PR", "RS", "SC"}:
        return "S" # South
    return None 


def group_age(age: str) -> str:
    """Converts age (attribute "idade_paciente") into a category based on
    a epidemiological (clinical) grouping."""
    if age is None:
        return None
    
    val_type, age_val = tuple(map(int, age.split("-")))

    if val_type == 3: # Months
        if age_val > 12:
            return None
        return "1" # Infant
    
    if val_type != 4 or age_val > 120:
        return None # Inconsistent or wrong data
    
    if age_val <= 4:
        return "2" # Young children
    if age_val <= 12:
        return "3" # Children
    if age_val <= 17:
        return "4" # Adolescents
    if age_val <= 49:
        return "5" # Young adulds
    if age_val <= 64:
        return "6" # Older adults
    return "7" # Elderly


def group_diagnosis_delay(diagnosis_delay: str) -> str:
    """Converts the symptoms-diagnosis delay (attribute "dias_sintomas_notificacao")
    into a category based on a clinical phase-based grouping."""
    try:
        diagnosis_delay = -1 * int(diagnosis_delay) # Negative value
    except Exception:
        return None
    
    if diagnosis_delay <= 3:
        return "1" # Early diagnosis
    if diagnosis_delay <= 7:
        return "2" # Critical phase diagnosis
    return "3" # Late diagnosis


def process_age_as_numeric(age: str) -> int:
    """Converts age (attribute "idade_paciente") into a numeric value in years.
    Invalid or inconsistent data is set to None."""
    if age is None:
        return None
    
    val_type, age_val = tuple(map(int, age.split("-")))

    if val_type == 3: # Months
        if age_val > 12:
            return None
        return 0
    
    if val_type != 4 or age_val > 120 or age_val < 1:
        return None # Inconsistent or wrong data
    return age_val


def process_diagnosis_delay_as_numeric(diagnosis_delay: str) -> int:
    """Converts the symptoms-diagnosis delay (attribute "dias_sintomas_notificacao")
    into a numeric value in days. Invalid data is set to None."""
    try:
        diagnosis_delay = -1 * int(diagnosis_delay) # Negative value
    except Exception:
        return None
    
    if diagnosis_delay < 0:
        return None # Invalid data
    return diagnosis_delay
    

def parse_args() -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description="Process SINAN data.")
    parser.add_argument('input', type=str, help='Input Parquet file path')
    parser.add_argument('-c', '--convert_categorical', action='store_true', 
                        help='Convert categorical attributes to numeric')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output CSV file path')
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()

    input_path = Path(args['input'])
    convert_categorical = args['convert_categorical']
    output_path = args['output']

    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        raise RuntimeError(f"Error reading input file {input_path}: {e}")
    
    if output_path is None:
        if convert_categorical:
            output_path = input_path.parent / (input_path.name + "numeric.csv")
        else:
            output_path = input_path.parent / (input_path.name + ".csv")

    df = df[df["sexo_paciente"].isin(["M", "F"])]
    df.loc[df["sexo_paciente"] == "M", "gestante_paciente"] = "6"
    df["gestante_paciente"] = df["gestante_paciente"].fillna("9")

    df["raca_cor_paciente"] = df["raca_cor_paciente"].fillna("9")

    # Remove rows with missing data
    df = df.dropna(axis=0, how="any", subset=list(PATIENT_ATTRS))

    # Remove non-infected cases or Chikungunya cases
    df = df[df["classificacao_final"].isin(["10", "11", "12"])]

    target_map = {
        "10": "low_risk", # 10 = Dengue
        "11": "alarm",    # 11 = Dengue com Sinais de Alarme
        "12": "severe"    # 12 = Dengue Grave
    }
    df["severity"] = df["classificacao_final"].map(target_map)

    df = df.drop("classificacao_final", axis=1)

    df["sigla_uf_residencia"] = df["sigla_uf_residencia"].apply(uf_to_region)
    df = df.dropna(axis=0, how="any", subset=["sigla_uf_residencia"])

    cols_to_keep = list(PATIENT_ATTRS) + ["severity"]
    df.drop(
        [col for col in df.columns if col not in cols_to_keep], 
        inplace=True, axis=1
    )

    # Process numeric columns
    if convert_categorical:
        df["idade_paciente"] = df["idade_paciente"].apply(process_age_as_numeric)
        df["dias_sintomas_notificacao"] = df["dias_sintomas_notificacao"].apply(process_diagnosis_delay_as_numeric)
        df = df.dropna(axis=0, how="any", subset=list(NUMERIC_ATTRS))
        for col in NUMERIC_ATTRS:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

        for col in BINARY_ATTRS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] == 2, col] = 0

        cols_to_encode = [col for col in df.columns if col in CATEGORICAL_ATTRS]
        prefix_map = {col: f'one_hot_{col}' for col in cols_to_encode}
        df = pd.get_dummies(df, columns=cols_to_encode, prefix=prefix_map, dtype=int)
    else:
        df["idade_paciente"] = df["idade_paciente"].apply(group_age)
        df["dias_sintomas_notificacao"] = df["dias_sintomas_notificacao"].apply(group_diagnosis_delay)
        df = df.dropna(axis=0, how="any", subset=list(NUMERIC_ATTRS))

    df.to_csv(output_path, encoding='utf-8', index=False)