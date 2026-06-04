# import packages
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import src.analysis as an
import numpy as np
import pandas as pd


# page configurtion
st.set_page_config(layout="wide")

# set title 
st.title("Rental Prices in the Czech Republic")

# about
st.header("About")

st.text("This is an interactive environment for the Data Processing in Python final project created by Matyáš Tvrz and Jonathan Eugenio Gaeta. " \
"We scrape data from sreality.cz and bezrealitky.cz on rental properties in the Czech Republic. " \
"In the first part, we present an interactive heatmap (choropleth) based on the median rental prices per meter squared in each region. " \
"The map includes popups with information about specific properties including a link to the public listing. " \
"The user can filter the properties based on characteristics and price. " \
"In the second part, we analyze the data. " \
"We allow the user to filter by cities, flat types, and area. " \
"First, we present basic summary statistics of the rental prices, grouped by districts and flat types. " \
"Next, we present exploratory plots, which visually illustrate some of the features of the data. " \
"Finally, we estimate two simple regressions: a level OLS on log-level OLS. " \
"We allow for the user to toggle controls for districts and flat types, and display the results, along with diagnostic plots and a fixed effects plot for districts." )


# heatmap
st.header("Heatmap of Rental Prices")

col1, col2 = st.columns([1, 3])

with col1:
    show_heatmap = st.checkbox("Show Heatmap", value=True)

    st.text("Filters:")

    property_layer = st.radio(
        "Property Source",
        ["All", "Sreality", "Bezrealitky", "None"],
        index=0
    )

    region_level = st.radio(
        "Administrative level",
        ["Kraje", "Okresy", "Obce"],
        index=0
    )

    # flat type filter
    standard_types = ['1+kk','1+1','2+kk','2+1','3+kk','3+1','4+kk','4+1','5+kk','5+1','atypické','other']
    selected_flat_types = st.multiselect(
        "Flat type",
        options=standard_types,
        default=standard_types
    )

    # area and price sliders
    df_max = pd.read_parquet("data/processed/df_heatmap.parquet")

    # set correct maximum (999th quantile for extreme outliers) values for sliders
    area_max = int(df_max['area'].quantile(0.999))
    price_max = int(df_max['price'].quantile(0.999))

    area_range = st.slider("Area (m²)", min_value=0, max_value=area_max, value=(0, area_max))
    price_range = st.slider("Total price per month (CZK)", min_value=0, max_value=price_max, step=1_000, value=(0, price_max))

from src.draw_heatmap import draw_heatmap_streamlit

with col2:
    m = draw_heatmap_streamlit(
        region_level=region_level,
        property_layer=property_layer,
        show_heatmap=show_heatmap,
        selected_flat_types=selected_flat_types,
        area_range=area_range,
        price_range=price_range
    )

    components.html(m._repr_html_(), height=800, scrolling=True)



# analysis
st.header("Analysis of Determinants")

df_reg = pd.read_csv("data/processed/df_reg.csv")

# filters in sidebar
st.subheader("Filters")

st.text("Use filters for the dataset which is used for the following analysis. The default option uses all properties (extreme observations with area beyond the 999th quantile are excluded).")

col1, col2, col3 = st.columns(3)

# filter cities based on name
with col1:
    all_cities = sorted(df_reg["city"].unique())
    selected_cities = st.multiselect(
        "Cities",
        options=all_cities,
        default=all_cities
    )

# filter flat types
with col2:
    all_flat_types = sorted(df_reg["flat_type"].unique())
    selected_flat_types = st.multiselect(
        "Flat types",
        options=all_flat_types,
        default=all_flat_types
    )

# slider for area (meters squared)
with col3:
    area_range = st.slider(
    "Area (m²)",
    min_value=0,
    max_value=area_max,
    value=(0, area_max),
    key="area_range_analysis"
)

# apply filters
df_filtered = df_reg[
    df_reg["city"].isin(selected_cities) &
    df_reg["flat_type"].isin(selected_flat_types) &
    df_reg["area"].between(*area_range)
]

# count of observations based on selected filter
st.caption(f"{len(df_filtered):,} listings selected out of {len(df_reg):,}")

# summary stats for filtered selection
st.subheader("Summary Statistics")

st.text("See the summary statistics for rental prices, grouped either by flat type or district, ordered by median price.")

df_filtered_stats = df_filtered.copy()
df_filtered_stats['price_m2'] = df_filtered_stats['price'] / df_filtered_stats['area']

tab1, tab2 = st.tabs(["By Flat Type", "By District"])

with tab1:
    by_flat = (
        df_filtered_stats.groupby('flat_type')
        .agg(
            count=('price', 'count'),
            median_price=('price', 'median'),
            mean_price=('price', 'mean'),
            median_price_m2=('price_m2', 'median'),
        )
        .round(0)
        .sort_values('median_price', ascending=False)
    )
    st.dataframe(by_flat, use_container_width=True)

with tab2:
    by_district = (
        df_filtered_stats.groupby('district')
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
    st.dataframe(by_district, use_container_width=True)

st.divider()


st.subheader("Exploratory Analysis")

st.text("See exploratory analysis plots: median rent by district of the top 15 districts by listing count, price per meter squared for main flat types, " \
"a simple correlation matrix of main variables, and rent vs. distance to Prague, colored by flat type (shows only 100 sampled properties per flat type, to reduce clutter).")

st.pyplot(an.plot_eda(df_filtered), use_container_width=False)

# toggles for model specifications - adding controls
st.subheader("Model Specification")

st.markdown(
    r"""
We estimate the model

$$
\text{Rent}_i=\beta_0+\beta_1\text{Area}_i+\beta_2\text{DistanceToPrague}_i
+\gamma' \text{FlatType}_i+\delta' \text{District}_i+\varepsilon_i.
$$

Here, you can choose to omit the district or flat-type controls:
"""
)

col_spec1, col_spec2 = st.columns(2)
with col_spec1:
    include_flat_type = st.checkbox("Control for flat type", value=True)
with col_spec2:
    include_district = st.checkbox("Control for district", value=True)

# run both models
if len(df_filtered) < 10:
    st.warning("Too few observations — adjust the filters.")

else:
    model_ols = an.run_ols(df_filtered, include_flat_type, include_district)
    model_log, baseline_district = an.run_log_ols(df_filtered, include_flat_type, include_district)

    st.divider()
    st.subheader("Model Results")

    col_ols, col_log = st.columns(2)

    with col_ols:
        st.markdown("**OLS (levels)**")
        st.metric("Area coefficient",     f"{model_ols.params['area']:.2f}",
                delta=f"{model_ols.params['area']:.2f} CZK per m²")
        st.metric("Distance coefficient", f"{model_ols.params['distance_prague_km']:.2f}",
                delta=f"{model_ols.params['distance_prague_km']:.2f} CZK per km")
        st.metric("Adjusted R²",          f"{model_ols.rsquared_adj:.3f}")

    with col_log:
        st.markdown("**Log-linear OLS**")
        st.metric("Area coefficient",     f"{model_log.params['area']:.4f}",
                delta=f"~{model_log.params['area']*100:.2f}% per m²")
        st.metric("Distance coefficient", f"{model_log.params['distance_prague_km']:.4f}",
                delta=f"~{model_log.params['distance_prague_km']*100:+.2f}% per km")
        st.metric("Adjusted R²",          f"{model_log.rsquared_adj:.3f}")

    st.divider()
    st.subheader("Diagnostics")
    st.pyplot(an.plot_diagnostics(df_filtered, model_ols), use_container_width=False)

    st.divider()
    st.subheader("District Fixed Effects")
    st.text("This plot displays how much each district raises or lowers rent relative to the baseline district, after controlling for covariates. " \
    "The baseline district is chosen as to have its median price closest to the overall median price. The plot displays the top and bottom 8 districts. ")
    if include_district:
        st.pyplot(an.plot_district_fe(model_log, baseline_district), use_container_width=False)
    else:
        st.info("Enable district controls to see the fixed effects plot.")

