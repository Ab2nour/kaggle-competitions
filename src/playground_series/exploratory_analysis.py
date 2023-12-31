import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_correlation_matrix(df_quanti: pd.DataFrame, threshold: float = 0.5) -> None:
    """Plots the correlation matrix of `df_quanti`,
    a DataFrame with only quantitative variables.

    Correlation values under `threshold` will be hidden.
    """
    plt.figure(figsize=(8, 8))

    df_corr = df_quanti.corr()

    sns.heatmap(
        df_corr[df_corr.abs() > threshold],
        annot=True,
        cmap="BrBG",
        linewidths=0.5,
        vmax=1,
        vmin=-1,
    )


def plot_na_quanti(df_quanti: pd.DataFrame) -> None:
    """Plots the missing values
    todo"""
    df_na = df_quanti.isna()
    df_na_sum = df_na.sum()
    nb_missing_values = df_na_sum.sum()

    if nb_missing_values == 0:
        print("There are no missing values.")
        return
    else:
        print(f"There are {nb_missing_values} missing values.")

    _, ax = plt.subplots(1, 2, figsize=(15, 4))

    ax[0].set_title("Number of missing values")
    sns.barplot(x=df_na_sum.values, y=df_na_sum.index, color="C0", ax=ax[0])

    ax[1].set_title("Missing values")
    sns.heatmap(df_na, cbar=False, ax=ax[1])

    plt.show()

    print(df_na_sum)


def plot_distributions_quanti(df_quanti: pd.DataFrame) -> None:
    for var in df_quanti.columns:
        _, ax = plt.subplots(1, 2, figsize=(8, 2))

        sns.boxplot(x=df_quanti[var], width=0.25, ax=ax[0])
        sns.histplot(df_quanti[var], kde=True, ax=ax[1])

        plt.show()


def plot_distributions_quali(df_quali: pd.DataFrame) -> None:
    for var in df_quali.columns:
        if df_quali[var].nunique() > 3:
            sns.histplot(y=df_quali[var])
        else:
            plt.figure(figsize=(4, 2))
            sns.histplot(df_quali[var], shrink=0.3)
    plt.show()
