import folium
from folium.plugins import MarkerCluster, GroupedLayerControl
import geopandas as gpd
import pandas as pd
from function_scripts import add_choropleth_layer
from function_scripts import add_properties
import webbrowser



def draw_heatmap(df_heatmap, df_sreality_property, df_bezrealitky_property):
    # base map, centered on CZ
    m = folium.Map(location=(49.75, 15.40), zoom_start = 8)

    okresy = gpd.read_file("data/okresy.json") 

    kraje = gpd.read_file("data/kraje.json") 

    obce = gpd.read_file("data/obce.json")

    # remove invalid areas
    df_heatmap = df_heatmap.dropna(subset=['area'])
    df_heatmap = df_heatmap[df_heatmap['area'] > 0]

    # price per m²
    df_heatmap['price_m2'] = (
        df_heatmap['price'] / df_heatmap['area']
    )

    # winsorization
    lower = df_heatmap['price_m2'].quantile(0.05)
    upper = df_heatmap['price_m2'].quantile(0.95)

    df_heatmap['price_m2_wins'] = (
        df_heatmap['price_m2']
        .clip(lower=lower, upper=upper)
    )

    # convert to GeoDataFrame
    gdf_properties = gpd.GeoDataFrame(
        df_heatmap,
        geometry=gpd.points_from_xy(
            df_heatmap['lon'],
            df_heatmap['lat']
        ),
        crs="EPSG:4326"
    )

    kraje_layer = add_choropleth_layer(
        m,
        kraje,
        gdf_properties,
        "Kraje"
    )

    okresy_layer = add_choropleth_layer(
        m,
        okresy,
        gdf_properties,
        "Okresy"
    )

    obce_layer = add_choropleth_layer(
        m,
        obce,
        gdf_properties,
        "Obce"
    )

    # realty website indicator
    df_sreality_property['source'] = 'Sreality'
    df_bezrealitky_property['source'] = 'Bezrealitky'

    # sreality property popups layer
    sreality_layer = folium.FeatureGroup(
        name="Sreality Properties",
        show=False
    )

    sreality_cluster = MarkerCluster().add_to(sreality_layer)

    # create popups with variables from df_sreality_property
    for _, row in df_sreality_property.iterrows():

        popup_html = f"""
        <b>{row['price']:,} CZK</b><br>
        <b>Source:</b> {row['source']}<br>
        {row['locality']}<br>
        {row['flat_type']}<br>
        {row['area']} m²<br><br>
        <img src="{row['image']}" width="200"><br>
        <a href="{row['url']}" target="_blank">Open listing</a>
        """

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=4,
            fill=True,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(sreality_cluster)

    sreality_layer.add_to(m)

    # bezrealitky property popups layer
    bezrealitky_layer = folium.FeatureGroup(
        name="Bezrealitky Properties",
        show=False
    )

    bezrealitky_cluster = MarkerCluster().add_to(bezrealitky_layer)

    # create popups with variables from df_bezrealitky_property
    for _, row in df_bezrealitky_property.iterrows():

        popup_html = f"""
        <b>{row['price']:,} CZK</b><br>
        <b>Source:</b> {row['source']}<br>
        {row['locality']}<br>
        {row['flat_type']}<br>
        {row['area']} m²<br><br>
        <img src="{row['image']}" width="200"><br>
        <a href="{row['url']}" target="_blank">Open listing</a>
        """

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=4,
            fill=True,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(bezrealitky_cluster)

    bezrealitky_layer.add_to(m)


    # all datasets merged property popups
    all_property = pd.concat(
        [df_sreality_property, df_bezrealitky_property],
        ignore_index=True
    )

    all_layer = folium.FeatureGroup(
        name="All Properties",
        show=True
    )

    all_cluster = MarkerCluster().add_to(all_layer)

    # create popups with variables from merged dataset
    for _, row in all_property.iterrows():

        popup_html = f"""
        <b>{row['price']:,} CZK</b><br>
        <b>Source:</b> {row['source']}<br>
        {row['locality']}<br>
        {row['flat_type']}<br>
        {row['area']} m²<br><br>
        <img src="{row['image']}" width="200"><br>
        <a href="{row['url']}" target="_blank">Open listing</a>
        """

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=4,
            fill=True,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(all_cluster)

    all_layer.add_to(m)

    # layer control
    folium.LayerControl(collapsed=False).add_to(m)

    GroupedLayerControl(
        groups={
            "Property Layers": [
                all_layer,
                sreality_layer,
                bezrealitky_layer
            ],
            "Choropleths": [
                kraje_layer,
                okresy_layer,
                obce_layer
            ]
        },
        exclusive_groups=True,
        collapsed=False
    ).add_to(m)

    path = "/tmp/heatmap.html"

    m.save(path)

    webbrowser.open(f"file://{path}")


def draw_heatmap_streamlit(df_heatmap, df_sreality_property, df_bezrealitky_property, region_level="Kraje", property_layer="All", show_heatmap=True):
    # base map, centered on CZ
    m = folium.Map(location=(49.75, 15.40), zoom_start = 8)

    okresy = gpd.read_file("data/okresy.json") 

    kraje = gpd.read_file("data/kraje.json") 

    obce = gpd.read_file("data/obce.json")

    # remove invalid areas
    df_heatmap = df_heatmap.dropna(subset=['area'])
    df_heatmap = df_heatmap[df_heatmap['area'] > 0]

    # price per m²
    df_heatmap['price_m2'] = (
        df_heatmap['price'] / df_heatmap['area']
    )

    # winsorization
    lower = df_heatmap['price_m2'].quantile(0.05)
    upper = df_heatmap['price_m2'].quantile(0.95)

    df_heatmap['price_m2_wins'] = (
        df_heatmap['price_m2']
        .clip(lower=lower, upper=upper)
    )

    # convert to GeoDataFrame
    gdf_properties = gpd.GeoDataFrame(
        df_heatmap,
        geometry=gpd.points_from_xy(
            df_heatmap['lon'],
            df_heatmap['lat']
        ),
        crs="EPSG:4326"
    )

    if show_heatmap:

        if region_level == "Kraje":
            add_choropleth_layer(m, kraje, gdf_properties, "Kraje")

        elif region_level == "Okresy":
            add_choropleth_layer(m, okresy, gdf_properties, "Okresy")

        elif region_level == "Obce":
            add_choropleth_layer(m, obce, gdf_properties, "Obce")

    # realty website indicator
    df_sreality_property['source'] = 'Sreality'
    df_bezrealitky_property['source'] = 'Bezrealitky'

    if property_layer == "All":

        all_df = pd.concat(
            [
                df_sreality_property.assign(source="Sreality"),
                df_bezrealitky_property.assign(source="Bezrealitky")
            ],
            ignore_index=True
        )

        add_properties(all_df, m)


    elif property_layer == "Sreality":

        add_properties(
            df_sreality_property.assign(source="Sreality"),
            m
        )


    elif property_layer == "Bezrealitky":

        add_properties(
            df_bezrealitky_property.assign(source="Bezrealitky"),
            m
        )


    return m