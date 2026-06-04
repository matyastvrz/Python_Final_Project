# ── imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import re
import ipywidgets as widgets
from IPython.display import display
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import plotly.graph_objects as go


# ── design system ─────────────────────────────────────────────────────────────
NAVY     = "#1d3557"
BLUE     = "#457b9d"
TEAL     = "#2a9d8f"
CORAL    = "#e63946"
BG       = "none"
BG_LIGHT = "none"
GRID_COL = "#2b2b2b"
TEXT     = "#f8f9fa"
MUTED    = "#868e96"

MAIN_TYPES  = ['1+kk', '1+1', '2+kk', '2+1', '3+kk', '3+1', '4+kk', '4+1']
TYPE_COLORS = dict(zip(MAIN_TYPES,
    ['#457b9d','#2a9d8f','#f4a261','#e63946','#6d6875','#b5838d','#1d3557','#52b788']))

plt.rcParams.update({
    'figure.facecolor'   : 'none',
    'axes.facecolor'     : 'none',
    'axes.grid'          : True,
    'grid.color'         : GRID_COL,
    'grid.linewidth'     : 0.8,
    'axes.axisbelow'     : True,
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'axes.spines.left'   : False,
    'axes.spines.bottom' : True,
    'xtick.color'        : TEXT,
    'ytick.color'        : TEXT,
    'axes.labelcolor'    : TEXT,
    'axes.titlecolor'    : TEXT,
    'axes.titlesize'     : 13,
    'axes.titleweight'   : 'bold',
    'axes.titlepad'      : 12,
    'axes.labelsize'     : 10,
    'font.size'          : 10,
    'text.color'         : TEXT,
    'legend.facecolor'   : BG_LIGHT,
    'legend.edgecolor'   : GRID_COL,
    'legend.framealpha'  : 0.9,
})

# ── helpers ───────────────────────────────────────────────────────────────────
def _fmt_kczk(ax, axis='x'):
    """
    Format axis tick labels as thousands of Czech koruna.

    Args:
        ax: Matplotlib axis object.
        axis (str): Which axis to format, either 'x' or 'y'.
    """
    fmt = mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k")
    if axis == 'x': ax.xaxis.set_major_formatter(fmt)
    else:           ax.yaxis.set_major_formatter(fmt)

def _main_df(df):
    """Keep only the 8 main flat types."""
    return df[df['flat_type'].isin(MAIN_TYPES)].copy()

def _sample_by_type(df, n=50):
    """Sample up to n rows per flat type for visualization.

    Args:
        df: DataFrame with a 'flat_type' column.
        n (int): Maximum rows to sample per flat type.

    Returns:
        pandas.DataFrame: Sampled rows across the main flat types.
    """
    parts = [
        g.sample(min(len(g), n), random_state=42)
        for ft in MAIN_TYPES
        for g in [df[df['flat_type'] == ft]]
        if len(g) >= 3
    ]
    return pd.concat(parts) if parts else df

def _big_title(fig, text, y=0.97):
    """Render a large centered title on a figure."""
    fig.text(0.5, y, text, ha='center', va='bottom',
             fontsize=15, fontweight='bold', color=TEXT)


# ── load data ─────────────────────────────────────────────────────────────────
def load_data(path="data/processed/df_reg.csv"):
    """Load the prepared regression dataset from CSV."""
    return pd.read_csv(path)


# ── summary stats by flat type ────────────────────────────────────────────────
def summary_stats_type(df_reg):
    """Compute summary statistics by flat type."""
    df = df_reg.copy()
    df['price_m2'] = df['price'] / df['area']
    return (
        df.groupby('flat_type')
          .agg(count          = ('price', 'count'),
               median_price   = ('price', 'median'),
               mean_price     = ('price', 'mean'),
               median_price_m2= ('price_m2', 'median'))
          .round(0)
          .sort_values('median_price', ascending=False)
    )


# ── interactive summary stats ─────────────────────────────────────────────────
def summary_stats_loc_interactive(df_reg):
    """Display interactive summary tables for cities and districts."""
    df = df_reg.copy()
    df['price_m2'] = df['price'] / df['area']
 
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
 
    cities    = ['All'] + sorted(df['city'].unique().tolist())
    dropdown  = widgets.Dropdown(options=cities, description='City:')
    out       = widgets.Output()
 
    def show(city):
        """Render the selected city summary and optional district breakdown."""
        out.clear_output(wait=True)
        with out:
            if city == 'All':
                display(
                    by_city.style
                    .format('{:,.0f}', subset=['median_price', 'mean_price',
                                               'median_price_m2', 'distance_prague_km'])
                    .set_caption('All cities')
                )
            else:
                display(
                    by_city.loc[[city]].style
                    .format('{:,.0f}', subset=['median_price', 'mean_price',
                                               'median_price_m2', 'distance_prague_km'])
                    .set_caption(f'{city} — overview')
                )
                if city in by_district.index.get_level_values('city'):
                    sub = by_district.loc[city]
                    if len(sub) > 1:
                        display(
                            sub.style
                            .format('{:,.0f}', subset=['median_price', 'mean_price',
                                                       'median_price_m2'])
                            .set_caption(f'{city} — by district')
                        )
 
    dropdown.observe(lambda change: show(change['new']), names='value')
    show('All')
    display(widgets.VBox([dropdown, out]))
 


# ── EDA — four-panel overview ─────────────────────────────────────────────────
def plot_eda(df_reg):
    """
    A  Median rent by district — top 15 markets
    B  Price per m² — violin by flat type
    C  Correlation matrix
    D  Rent vs Distance to Prague — coloured by flat type
    """
    df   = df_reg.copy()
    df_m = _main_df(df)
    df_m['price_per_m2'] = df_m['price'] / df_m['area']

    # top 15 districts by listing count
    top15 = df['district'].value_counts().head(15).index.tolist()

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG)
    gs  = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.35,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)

    # ── A: median rent by district ─────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    med  = (df[df['district'].isin(top15)]
              .groupby('district')['price']
              .median()
              .sort_values())
    pivot = med.median()
    clrs  = [TEAL if v >= pivot else MUTED for v in med]
    ax_a.barh(med.index, med.values / 1000, color=clrs, edgecolor=BG, height=0.7)
    ax_a.axvline(pivot / 1000, color=CORAL, lw=1.4, ls='--',
                 label=f'median  {pivot/1000:.0f} k')
    for i, (nm, val) in enumerate(med.items()):
        ax_a.text(val / 1000 + 0.3, i, f'{val/1000:.0f}',
                  va='center', fontsize=8, color=TEXT)
    ax_a.set_xlabel('Median rent (CZK thousands)')
    ax_a.set_title('Median Rent by District')
    ax_a.legend(fontsize=9)
    ax_a.tick_params(axis='y', labelsize=8.5)
    ax_a.grid(axis='x')
    ax_a.grid(axis='y', alpha=0)

    # ── B: price/m² violin by flat type ───────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    order   = (df_m.groupby('flat_type')['price_per_m2']
                   .median().sort_values(ascending=False).index.tolist())
    palette = [TYPE_COLORS[t] for t in order]
    sns.violinplot(data=df_m, x='flat_type', y='price_per_m2',
                   order=order, palette=palette,
                   inner='box', linewidth=0.8, cut=0.5, ax=ax_b)
    ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax_b.set_xlabel('Flat type')
    ax_b.set_ylabel('CZK per m²')
    ax_b.set_title('Price per m² by Flat Type')
    ax_b.grid(axis='y')
    ax_b.grid(axis='x', alpha=0)

    # ── C: correlation matrix ──────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    df['price_per_m2'] = df['price'] / df['area']
    existing = [c for c in ['price', 'area', 'distance_prague_km', 'price_per_m2']
                if c in df.columns]
    corr = df[existing].corr()
    labels = {'price': 'Rent', 'area': 'Area (m²)',
               'distance_prague_km': 'Dist. to Prague', 'price_per_m2': 'Price / m²'}
    corr.index   = [labels.get(c, c) for c in corr.index]
    corr.columns = [labels.get(c, c) for c in corr.columns]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask,
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                vmin=-1, vmax=1, annot=True, fmt='.2f',
                linewidths=1, linecolor=BG,
                annot_kws={'size': 11, 'weight': 'bold'},
                ax=ax_c, cbar_kws={'shrink': 0.75, 'pad': 0.02})
    ax_c.set_title('Correlation Matrix')
    ax_c.tick_params(labelsize=10)
    ax_c.set_facecolor(BG)

    # ── D: rent vs distance, coloured by flat type (100 pts/type) ──────────
    ax_d = fig.add_subplot(gs[1, 1])
    sample = _sample_by_type(df_m, n=100)
    for ft in MAIN_TYPES:
        grp = sample[sample['flat_type'] == ft]
        if grp.empty: continue
        ax_d.scatter(grp['distance_prague_km'], grp['price'] / 1000,
                     alpha=0.6, s=25, label=ft,
                     color=TYPE_COLORS[ft], linewidths=0)
    x = df_m['distance_prague_km'].values
    y = df_m['price'].values / 1000
    sl, ic, *_ = stats.linregress(x, y)
    xl = np.linspace(x.min(), x.max(), 200)
    ax_d.plot(xl, ic + sl * xl, color=CORAL, lw=2, ls='--',
              zorder=5, label='OLS trend')
    ax_d.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}k'))
    ax_d.set_xlabel('Distance to Prague (km)')
    ax_d.set_ylabel('Rent (CZK thousands)')
    ax_d.set_title('Rent vs Distance to Prague')
    ax_d.legend(fontsize=8, markerscale=1.4, ncol=2, loc='upper right')
    ax_d.grid()

    _big_title(fig, 'Czech Rental Market — Exploratory Analysis')
    plt.close()
    return fig


# ── OLS (levels) ──────────────────────────────────────────────────────────────
def run_ols(df_reg, include_flat_type=True, include_district=True):
    """Fit an OLS regression model for rent levels.

    Args:
        df_reg: Prepared regression DataFrame.
        include_flat_type (bool): Include flat type fixed effects.
        include_district (bool): Include district fixed effects.

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: Fitted model.
    """
    base = 'price ~ area + distance_prague_km'
    if include_flat_type: base += ' + C(flat_type)'
    if include_district:
        median_price = df_reg['price'].median()
        district_medians = df_reg.groupby('district')['price'].median()
        baseline_district = (district_medians - median_price).abs().idxmin()
        base += f" + C(district, Treatment('{baseline_district}'))"
    model = smf.ols(formula=base, data=df_reg).fit()
    print(model.summary())
    a = model.params['area']
    d = model.params['distance_prague_km']
    print(f'\n── Coefficient interpretation ──────────────────────────────')
    print(f'  area            : {a:.4f}  →  +{a:.2f}Kč rent per extra m²')
    print(f'  dist. to Prague : {d:.4f}  →  {d:+.2f}Kč rent per extra km')
    print(f'────────────────────────────────────────────────────────────')
    return model


# ── diagnostic plots ──────────────────────────────────────────────────────────
def plot_diagnostics(df_reg, model):
    """
    Three-panel OLS diagnostics:
      Residuals vs Fitted · Normal Q-Q · Rent Distribution
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.86,
                        bottom=0.14, wspace=0.34)

    std_resid = model.get_influence().resid_studentized_internal

    # residuals vs fitted
    axes[0].scatter(model.fittedvalues, model.resid,
                    alpha=0.2, s=12, color=BLUE, linewidths=0)
    axes[0].axhline(0, color=CORAL, lw=1.2, ls='--')
    sm_pts = lowess(model.resid, model.fittedvalues, frac=0.35)
    axes[0].plot(sm_pts[:, 0], sm_pts[:, 1], color=CORAL, lw=2, label='LOWESS')
    axes[0].legend(fontsize=9)
    _fmt_kczk(axes[0], 'y')
    _fmt_kczk(axes[0], 'x')
    axes[0].set_xlabel('Fitted values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted')

    # Q-Q plot
    (osm, osr), (slope, intercept, _) = stats.probplot(std_resid)
    axes[1].scatter(osm, osr, alpha=0.3, s=12, color=BLUE, linewidths=0)
    axes[1].plot(osm, slope * np.array(osm) + intercept,
                 color=CORAL, lw=1.8, ls='--')
    axes[1].set_xlabel('Theoretical quantiles')
    axes[1].set_ylabel('Std. residuals')
    axes[1].set_title('Normal Q-Q Plot')

    # rent distribution + log-normal
    prices = df_reg['price']
    axes[2].hist(prices / 1000, bins=50,
                 color=BLUE, edgecolor=BG, alpha=0.80, density=True)
    mu, sigma = np.log(prices).mean(), np.log(prices).std()
    xs  = np.linspace(prices.min(), prices.max(), 300)
    pdf = stats.lognorm.pdf(xs, s=sigma, scale=np.exp(mu))
    axes[2].plot(xs / 1000, pdf * 1000, color=CORAL, lw=2.2, label='Log-normal fit')
    axes[2].legend(fontsize=9)
    axes[2].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:.0f}k'))
    axes[2].set_xlabel('Rent (CZK thousands)')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Rent Distribution')

    _big_title(fig, 'OLS Model Diagnostics')
    plt.close()
    return fig


# ── log-OLS ───────────────────────────────────────────────────────────────────
def run_log_ols(df_reg, include_flat_type=True, include_district=True):
    """
    Fit a log-level OLS regression model for rent.

    Args:
        df_reg: Prepared regression DataFrame.
        include_flat_type (bool): Include flat type fixed effects.
        include_district (bool): Include district fixed effects.

    Returns:
        tuple: (fitted model, baseline district label).
    """
    baseline_district = None
    df_reg = df_reg.copy()
    df_reg['log_price'] = np.log(df_reg['price'])
    base = 'log_price ~ area + distance_prague_km'
    if include_flat_type: base += ' + C(flat_type)'
    if include_district:
        median_price = df_reg['price'].median()
        district_medians = df_reg.groupby('district')['price'].median()
        baseline_district = (district_medians - median_price).abs().idxmin()
        base += f" + C(district, Treatment('{baseline_district}'))"
    model = smf.ols(formula=base, data=df_reg).fit()

    print(model.summary())
    a = model.params['area']
    d = model.params['distance_prague_km']
    print(f'\n── Coefficient interpretation ──────────────────────────────')
    print(f'  area            : {a:.4f}  →  +{a*100:.2f}% rent per extra m²')
    print(f'  dist. to Prague : {d:.4f}  →  {d*100:+.2f}% rent per extra km')
    print(f'────────────────────────────────────────────────────────────')
    return model, baseline_district


# ── district fixed effects ────────────────────────────────────────────────────
def plot_district_fe(model_log, baseline_district):
    """
    Top 8 premium and top 8 discount districts, with 95% CI bars.
    """
    clean = lambda s: re.sub(r"C\(district,\s*Treatment\('[^']*'\)\)\[T\.", "", s).replace("]", "")

    fe    = model_log.params.filter(regex=r"C\(district")
    fe_ci = model_log.conf_int().filter(regex=r"C\(district", axis=0)
    fe.index    = fe.index.map(clean)
    fe_ci.index = fe_ci.index.map(clean)

    fe_show = pd.concat([fe.nsmallest(8), fe.nlargest(8)]).sort_values()
    ci_low  = fe_ci.loc[fe_show.index, 0]
    ci_high = fe_ci.loc[fe_show.index, 1]
    colors  = [TEAL if v >= 0 else CORAL for v in fe_show]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.22, right=0.93, top=0.88, bottom=0.10)

    ax.barh(fe_show.index, fe_show.values,
            color=colors, edgecolor=BG, height=0.65, alpha=0.88)
    ax.errorbar(fe_show.values, fe_show.index,
                xerr=[fe_show - ci_low, ci_high - fe_show],
                fmt='none', color='#495057', lw=1.1, capsize=3.5)
    ax.axvline(0, color=TEXT, lw=1.0, ls='--', alpha=0.5)

    for name, val in fe_show.items():
        pad = 0.005 if val >= 0 else -0.005
        ax.text(val + pad, name, f'{val:+.3f}',
                va='center', ha='left' if val >= 0 else 'right',
                fontsize=8.5, color=TEXT)

    ax.legend(handles=[
        mpatches.Patch(color=TEAL,  label='Rent premium'),
        mpatches.Patch(color=CORAL, label='Rent discount'),
    ], fontsize=9, loc='lower right')

    ax.set_xlabel('Log-price premium vs. baseline district')
    ax.set_title(f'District Fixed Effects — Top 8 and Bottom 8\n(Baseline: {baseline_district})')
    ax.set_facecolor(BG_LIGHT)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='x')
    ax.grid(axis='y', alpha=0)
    plt.close()
    return fig

