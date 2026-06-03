# import packages
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
 
# ── shared style constants ───────────────────────────────────────────────────
PALETTE  = "Set2"
ACCENT   = "#2a9d8f"
RED      = "#e76f51"
BG       = "#f8f9fa"
TITLE_FS = 13
LABEL_FS = 11
 
def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title,   fontsize=TITLE_FS, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
 
# ── load data ────────────────────────────────────────────────────────────────
def load_data(path="data/df_reg.csv"):
    df_reg = pd.read_csv(path)
    return df_reg
 
# ── summary stats by flat type ───────────────────────────────────────────────
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
 
# ── interactive summary stats by city / district ─────────────────────────────
def summary_stats_loc_interactive(df_reg):
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
 
# ── EDA visualizations ───────────────────────────────────────────────────────
def plot_eda(df_reg):
    """
    Four-panel EDA figure:
      A – Median rent by district (ranked horizontal bar)
      B – Price per m² by flat type (boxplot)
      C – Correlation matrix of numeric variables
      D – Rent vs Distance to Prague, coloured by flat type
    """
    df = df_reg.copy()
    df["price_per_m2"] = df["price"] / df["area"]
 
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("white")
    gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35)
 
    # ── A: median price by district ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    med  = df.groupby("district")["price"].median().sort_values(ascending=True)
    overall_med = med.median()
    colors = [ACCENT if v >= overall_med else "#adb5bd" for v in med]
    bars   = ax_a.barh(med.index, med.values / 1000,
                       color=colors, edgecolor="white", height=0.7)
    ax_a.axvline(overall_med / 1000, color=RED, lw=1.4, ls="--",
                 label=f"overall median ({overall_med/1000:.0f}k)")
    ax_a.legend(fontsize=9)
    for bar, val in zip(bars, med.values):
        ax_a.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                  f"{val/1000:.0f}k", va="center", fontsize=8)
    _style_ax(ax_a, "Median Rent by District",
              "Median rent (CZK thousands)", "")
 
    # ── B: price/m² boxplot by flat type ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    order = (df.groupby("flat_type")["price_per_m2"]
               .median()
               .sort_values(ascending=False)
               .index.tolist())
    palette = sns.color_palette(PALETTE, n_colors=len(order))
    sns.boxplot(data=df, x="flat_type", y="price_per_m2",
                order=order, palette=palette,
                flierprops=dict(marker=".", markersize=3, alpha=0.4),
                ax=ax_b)
    ax_b.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    _style_ax(ax_b, "Price per m² by Flat Type", "Flat type", "CZK / m²")
 
    # ── C: correlation matrix ─────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    num_cols = ["price", "area", "distance_prague_km", "price_per_m2"]
    existing = [c for c in num_cols if c in df.columns]
    corr     = df[existing].corr()
    labels   = {"price": "Rent", "area": "Area (m²)",
                "distance_prague_km": "Dist. Prague (km)",
                "price_per_m2": "Price / m²"}
    corr.index   = [labels.get(c, c) for c in corr.index]
    corr.columns = [labels.get(c, c) for c in corr.columns]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=0.5,
                annot_kws={"size": 10}, ax=ax_c,
                cbar_kws={"shrink": 0.8})
    ax_c.set_title("Correlation Matrix", fontsize=TITLE_FS,
                   fontweight="bold", pad=10)
    ax_c.tick_params(labelsize=9)
 
    # ── D: price vs distance, coloured by flat type ───────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    flat_types = df["flat_type"].unique()
    pal_d = dict(zip(flat_types,
                     sns.color_palette(PALETTE, n_colors=len(flat_types))))
    for ft, grp in df.groupby("flat_type"):
        ax_d.scatter(grp["distance_prague_km"], grp["price"] / 1000,
                     alpha=0.35, s=15, label=ft, color=pal_d[ft])
    x      = df["distance_prague_km"].values
    y      = df["price"].values / 1000
    slope, intercept, *_ = stats.linregress(x, y)
    xline  = np.linspace(x.min(), x.max(), 200)
    ax_d.plot(xline, intercept + slope * xline,
              color=RED, lw=1.8, ls="--", zorder=5, label="OLS trend")
    ax_d.legend(fontsize=8, markerscale=1.5, framealpha=0.7)
    ax_d.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}k"))
    _style_ax(ax_d, "Rent vs Distance to Prague",
              "Distance to Prague (km)", "Rent (CZK thousands)")
 
    fig.suptitle("Czech Rental Market — Exploratory Analysis",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.close()
    return fig

    
# ── Diagnostic plots ──────────────────────────────────────────────────────────
def plot_diagnostics(df_reg, model):
    """
    Three-panel diagnostic figure:
      • Residuals vs Fitted  (with LOWESS smoother)
      • Normal Q-Q of standardised residuals
      • Rent distribution with log-normal overlay
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("white")
 
    std_resid = model.get_influence().resid_studentized_internal
 
    # residuals vs fitted
    axes[0].scatter(model.fittedvalues, model.resid,
                    alpha=0.25, s=10, color=ACCENT)
    axes[0].axhline(0, color=RED, lw=1.2)
    sm = lowess(model.resid, model.fittedvalues, frac=0.3)
    axes[0].plot(sm[:, 0], sm[:, 1], color=RED, lw=1.8, label="LOWESS")
    axes[0].legend(fontsize=9)
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    _style_ax(axes[0], "Residuals vs Fitted", "Fitted values", "Residuals")
 
    # Q-Q plot
    (osm, osr), (slope, intercept, _) = stats.probplot(std_resid)
    axes[1].scatter(osm, osr, alpha=0.35, s=10, color=ACCENT)
    axes[1].plot(osm, slope * np.array(osm) + intercept,
                 color=RED, lw=1.5, ls="--")
    _style_ax(axes[1], "Normal Q-Q Plot",
              "Theoretical quantiles", "Std. residuals")
 
    # price distribution with log-normal fit
    prices = df_reg["price"]
    axes[2].hist(prices / 1000, bins=50,
                 color=ACCENT, edgecolor="white", alpha=0.85, density=True)
    mu, sigma = np.log(prices).mean(), np.log(prices).std()
    xs  = np.linspace(prices.min(), prices.max(), 300)
    pdf = stats.lognorm.pdf(xs, s=sigma, scale=np.exp(mu))
    axes[2].plot(xs / 1000, pdf * 1000, color=RED, lw=2, label="Log-normal fit")
    axes[2].legend(fontsize=9)
    axes[2].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    _style_ax(axes[2], "Rent Distribution", "Rent (CZK thousands)", "Density")
 
    fig.suptitle("OLS Diagnostics", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.close()
    return fig
 

# ── District fixed effects ────────────────────────────────────────────────────
def plot_district_fe(model_log):
    """
    Horizontal bar chart of district FE with 95% CI error bars,
    coloured by sign (premium / discount vs. baseline district).
    """
    clean = lambda s: s.replace("C(district)[T.", "").replace("]", "")
 
    fe_params = model_log.params.filter(like="C(district)")
    fe_conf   = model_log.conf_int().filter(like="C(district)", axis=0)
    fe_params.index = fe_params.index.map(clean)
    fe_conf.index   = fe_conf.index.map(clean)
 
    fe_sorted = fe_params.sort_values(ascending=True)
    ci_low    = fe_conf.loc[fe_sorted.index, 0]
    ci_high   = fe_conf.loc[fe_sorted.index, 1]
    colors    = [ACCENT if v >= 0 else RED for v in fe_sorted]
 
    fig, ax = plt.subplots(figsize=(10, max(5, len(fe_sorted) * 0.32)))
    fig.patch.set_facecolor("white")
 
    ax.barh(fe_sorted.index, fe_sorted.values,
            color=colors, edgecolor="white", height=0.65, alpha=0.85)
    ax.errorbar(fe_sorted.values, fe_sorted.index,
                xerr=[fe_sorted - ci_low, ci_high - fe_sorted],
                fmt="none", color="#495057", lw=1, capsize=3)
    ax.axvline(0, color="#333333", lw=1.0, ls="--")
 
    for name, val in fe_sorted.items():
        ax.text(val + (0.005 if val >= 0 else -0.005), name,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=7.5)
 
    _style_ax(ax, "District Fixed Effects (log-OLS)",
              "Log-price premium / discount vs. baseline district", "")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.close()
    return fig
 