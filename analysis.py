import pandas as pd 
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_data(path="data/df_reg.csv"):
    df_reg = pd.read_csv(path)
    return df_reg


def run_ols(df_reg):
    model = smf.ols(
        formula='price ~ area + C(flat_type) + C(district)',
        data=df_reg
    ).fit()

    print(model.summary())
    return model


def plot_diagnostics(df_reg, model):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # residuals vs fitted
    axes[0].scatter(model.fittedvalues, model.resid, alpha=0.3, s=10)
    axes[0].axhline(0, color='red', lw=1)
    axes[0].set_xlabel('Fitted'); axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted')

    # price distribution
    sns.histplot(df_reg['price'], bins=50, ax=axes[1])
    axes[1].set_title('Price Distribution')

    # price vs area
    axes[2].scatter(df_reg['area'], df_reg['price'], alpha=0.3, s=10)
    axes[2].set_xlabel('Area (m²)'); axes[2].set_ylabel('Price (CZK)')
    axes[2].set_title('Price vs Area')

    plt.tight_layout()
    plt.close()
    return fig


def run_log_ols(df_reg):
    df_reg['log_price'] = np.log(df_reg['price'])

    model_log = smf.ols(
        formula='log_price ~ area + C(flat_type) + C(district)',
        data=df_reg
    ).fit()

    print(model_log.summary())

    # Coefficient interpretation: area coef ≈ % change in price per extra m²
    print(f"\nArea coefficient: {model_log.params['area']:.4f}")
    print(f"→ Each extra m² is associated with ~{model_log.params['area']*100:.2f}% higher rent")

    return model_log


def plot_district_fe(model_log):
    fe = model_log.params.filter(like='C(district)')
    fe.index = fe.index.str.replace(r'C\(district\)\[T\.', '', regex=True).str.replace(']', '')
    fe_sorted = fe.sort_values(ascending=False)

    fig = plt.figure(figsize=(10, max(4, len(fe_sorted)*0.3)))
    fe_sorted.plot(kind='barh')
    plt.axvline(0, color='red', lw=1)
    plt.xlabel('Log-price premium vs baseline district')
    plt.title('District Fixed Effects')
    plt.tight_layout()
    plt.close()
    return fig