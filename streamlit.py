# import packages
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import analysis as an
import numpy as np
import pandas as pd


# page configurtion
st.set_page_config(layout="wide")

# set title 
st.title("Rental Prices in the Czech Republic")

# about
st.header("About")

st.text("This is an interactive environment for the Data Processing in Python final project.")


# heatmap
st.header("Heatmap of Rental Prices")

col1, col2 = st.columns([1, 3])

with col1:
    show_heatmap = st.checkbox("Show Heatmap", value=True)

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

    st.divider()

    # flat type filter
    standard_types = ['1+kk','1+1','2+kk','2+1','3+kk','3+1','4+kk','4+1','5+kk','5+1','atypické','other']
    selected_flat_types = st.multiselect(
        "Flat type",
        options=standard_types,
        default=standard_types
    )

    # area and price sliders
    df_max = pd.read_parquet("data/df_heatmap.parquet")

    # set correct maximum (999th quantile for extreme outliers) values for sliders
    area_max = int(df_max['area'].quantile(0.999))
    price_max = int(df_max['price'].quantile(0.999))

    area_range = st.slider("Area (m²)", min_value=0, max_value=area_max, value=(0, area_max))
    price_range = st.slider("Total price per month (CZK)", min_value=0, max_value=price_max, step=1_000, value=(0, price_max))

from draw_heatmap import draw_heatmap_streamlit

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
st.header("Analysis od Determinants")

df_reg = pd.read_csv("data/df_reg.csv")

# filters in sidebar
st.subheader("Filters")

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

# toggle between standard ols and log ols
model_choice = st.radio(
    "Model",
    ["OLS (levels)", "Log-linear"],
    horizontal=True
)

# filtered df must have atleast 10 observations
if len(df_filtered) < 10:
    st.warning("Too few observations — adjust the filters.")

# run ols or log ols based on toggle, display results and plots
else:
    if model_choice == "OLS (levels)":
        model = an.run_ols(df_filtered)
        fig_diag = an.plot_diagnostics(df_filtered, model)
        st.pyplot(fig_diag)

    else:
        model = an.run_log_ols(df_filtered)
        
        col_interp, col_fit = st.columns([1, 1])
        with col_interp:
            st.metric(
                "Area coefficient",
                f"{model.params['area']:.4f}",
                delta=f"~{model.params['area']*100:.2f}% per m²"
            )
        with col_fit:
            st.metric("R²", f"{model.rsquared:.3f}")

        fig_fe = an.plot_district_fe(model)
        st.pyplot(fig_fe)
