import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


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


def collect_correlated_variables(df, threshold=0.4):
    correlated = []
    n_cols = df.shape[1]
    for i in range(n_cols):
        col_i = df.columns[i]
        for j in range(i + 1, n_cols):
            col_j = df.columns[j]
            contingency_table = pd.crosstab(df[col_i], df[col_j], dropna=True)
            v = cramers_v(contingency_table)
            if (v > threshold):
                correlated.append((col_i, col_j, v))
    return correlated


def make_correlation_matrix(df):
    cols = df.columns
    n_cols = len(cols)
    corr_matrix = np.zeros((n_cols, n_cols))

    for i in range(n_cols):
        col_i = cols[i]
        for j in range(n_cols):
            col_j = cols[j]
            contingency_table = pd.crosstab(df[col_i], df[col_j], dropna=True)
            v = cramers_v(contingency_table)
            corr_matrix[i, j] = v

    return pd.DataFrame(corr_matrix, index=cols, columns=cols)


def plot_correlation_matrix(corr_matrix, output_path=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Cramér's V Correlation Heatmap")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def print_data_info(df):
    print("--- Data Info ---")
    df.info()

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Numerical Summary ---")
    print(df.describe(include = 'all'))

    missing_percentage = (df.isnull().sum() / len(df)) * 100
    print("--- Missing Value Percentage ---")
    print(missing_percentage.sort_values(ascending=False))


if __name__ == "__main__":
    import sys

    input_path = sys.argv[1]

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', None)

    df = pd.read_csv(input_path)

    print_data_info(df)

    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how="any", subset=["classificacao_final"])

    corr_matrix = make_correlation_matrix(df)
    plot_correlation_matrix(corr_matrix, output_path="correlation_heatmap.png")

    # correlated = collect_correlated_variables(df, threshold=0.4)
    # print("\n--- Highly Correlated Variables (Cramér's V > 0.4) ---")
    # for var1, var2, v in correlated:
    #     print(f"{var1} - {var2}: {v:.4f}")

