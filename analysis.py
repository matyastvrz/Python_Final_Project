# import packages
import pandas as pd 
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# load data - df_reg from files
def load_data(path="data/df_reg.csv"):
    df_reg = pd.read_csv(path)
    return df_reg

# get summary stats grouped by flat type 
def summary_stats_type(df_reg):
    df = df_reg.copy()
    df['price_m2'] = df['price'] / df['area']

    type_stats = (
        df.groupby('flat_type')
        .agg(
            count=('price', 'count'),
            median_price=('price', 'median'),
            mean_price=('price', 'mean'),
            median_price_m2=('price_m2', 'median'),
        )
        .round(0)
        .sort_values('median_price', ascending=False)
    )

    return type_stats

# get sumary stats grouped by city and districts
def summary_stats_loc_interactive(df_reg):
    df = df_reg.copy()
    df['price_m2'] = df['price'] / df['area']

    fmt = '{:,.0f}'.format

    # city-level summary
    by_city = (
        df.groupby('city')
        .agg(
            count=('price', 'count'),
            median_price=('price', 'median'),
            mean_price=('price', 'mean'),
            median_price_m2=('price_m2', 'median'),
            distance_prague_km=('distance_prague_km', 'mean'),
        )
        .round(0)
        .sort_values('median_price', ascending=False)
    )

    # district-level summary
    by_district = (
        df.groupby(['city', 'district'])
        .agg(
            count=('price', 'count'),
            median_price=('price', 'median'),
            mean_price=('price', 'mean'),
            median_price_m2=('price_m2', 'median'),
        )
        .round(0)
    )

    # create drop-down list option for cities using widgets
    cities = ['All'] + sorted(df['city'].unique().tolist())
    dropdown = widgets.Dropdown(options=cities, description='City:')
    out = widgets.Output()

    def show(city):
        out.clear_output(wait=True)
        with out:
            if city == 'All':
                display(
                    by_city.style
                    .format('{:,.0f}', subset=['median_price','mean_price','median_price_m2','distance_prague_km'])
                    .set_caption('All cities')
                )
            else:
                # city-level row
                display(
                    by_city.loc[[city]].style
                    .format('{:,.0f}', subset=['median_price','mean_price','median_price_m2','distance_prague_km'])
                    .set_caption(f'{city} — overview')
                )
                # district breakdown (only shown if city has sub-districts)
                if city in by_district.index.get_level_values('city'):
                    sub = by_district.loc[city]
                    if len(sub) > 1:
                        display(
                            sub.style
                            .format('{:,.0f}', subset=['median_price','mean_price','median_price_m2'])
                            .set_caption(f'{city} — by district')
                        )

    dropdown.observe(lambda change: show(change['new']), names='value')
    show('All')
    display(widgets.VBox([dropdown, out]))

# run standard OLS on explanatory variables
def run_ols(df_reg):
    model = smf.ols(
        formula='price ~ area + distance_prague_km + C(flat_type) + C(district)',
        data=df_reg
    ).fit()

    print(model.summary())
    return model

# show diagnostic plots 
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

# run OLS in log form 
def run_log_ols(df_reg):
    df_reg['log_price'] = np.log(df_reg['price'])

    model_log = smf.ols(
        formula='log_price ~ area + distance_prague_km + C(flat_type) + C(district)',
        data=df_reg
    ).fit()

    print(model_log.summary())

    # Coefficient interpretation: area coef ≈ % change in price per extra m²
    print(f"\nArea coefficient: {model_log.params['area']:.4f}")
    print(f"→ Each extra m² is associated with ~{model_log.params['area']*100:.2f}% higher rent")

    return model_log

# plot district fixed effects
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