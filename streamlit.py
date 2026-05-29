# import packages
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster


# page configurtion
st.set_page_config(layout="wide")

# set title 
st.title("Rental Price Determinants in the Czech Republic")

# about
st.header("About")

st.text("This is an interactive environment for the Data Processing in Python final project.")




# heatmap
st.header("Heatmap of Rental Prices")

# load data for heatmap from saved parquet files using load_data function
from function_scripts import load_data

(df_heatmap, df_sreality_property, df_bezrealitky_property) = load_data()


# layer control checkboxes
col1, col2 = st.columns([1, 3])

with col1:

    show_heatmap = st.checkbox(
        "Show Heatmap",
        value=True
    )

    property_layer = st.radio(
        "Property Layer",
        [
            "All",
            "Sreality",
            "Bezrealitky",
            "None"
        ],
        index=0
    )

    region_level = st.radio(
    "Administrative level",
    ["Kraje", "Okresy", "Obce"],
    index=0  
    )


from draw_heatmap import draw_heatmap_streamlit

m = draw_heatmap_streamlit(
    df_heatmap,
    df_sreality_property,
    df_bezrealitky_property,
    region_level=region_level,
    property_layer=property_layer,
    show_heatmap=show_heatmap
)

html_data = m._repr_html_()

components.html(
    html_data,
    height=800,
    scrolling=True
)
