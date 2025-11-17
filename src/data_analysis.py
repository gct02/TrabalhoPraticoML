import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from data_processing import process_data, read_parquet_data
from utils.constants import (
    PATIENT_ATTRS, PATIENT_DISEASES, PATIENT_SYMPTOMS
)

CORRELATIONS_TO_CHECK = [
    (col, "severity") for col in PATIENT_ATTRS
] + [
    (col1, col2) for col1 in PATIENT_DISEASES for col2 in PATIENT_SYMPTOMS
] + [
    ("prova_laco", col) for col in PATIENT_SYMPTOMS
]

def cramers_v(contingency_table):
    """ Calculate Cramér's V statistic for categorical-categorical association. """
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2_corr / (min((k_corr-1), (r_corr-1)) + 1e-10))


def compute_cramers_v(df):
    cramers_v_results = {}
    for col1, col2 in CORRELATIONS_TO_CHECK:
        contingency_table = pd.crosstab(df[col1], df[col2], dropna=True)
        v = cramers_v(contingency_table)
        cramers_v_results[(col1, col2)] = v
    return cramers_v_results


def collect_correlated_columns(cramers_v_results, threshold=0.4):
    return [(col1, col2, v) for (col1, col2), v in cramers_v_results.items() if v > threshold]


if __name__ == "__main__":
    import sys

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', None)

    df = read_parquet_data(input_path)

    print("=== Original Data Info ===")
    df.info()

    df = process_data(df, as_nominal=True)

    print("\n=== Processed Data Info ===")
    df.info()

    cramers_v_results = compute_cramers_v(df)

    cramers_v_results = sorted(
        cramers_v_results.items(), key=lambda item: item[1], reverse=True
    )
    cramers_v_results = {k: v for k, v in cramers_v_results}

    if output_path:
        with open(output_path, 'w') as f:
            f.write("Cramér's V Correlation Results:\n")
            for (col1, col2), v in cramers_v_results.items():
                f.write(f"{col1} - {col2}: {v:.4f}\n")
    else:
        print("\nCramér's V Correlation Results:")
        for (col1, col2), v in cramers_v_results.items():
            print(f"{col1} - {col2}: {v:.4f}")
