# import packages
import pandas as pd
import json 
import os
import requests  
import time
import re 
import random
from geopy.distance import geodesic

def process_data(update = False):

    if update == False:
        # get sreality df from last request
        df_sreality = pd.read_json("data/raw/df_sreality.json")
        df_sreality = pd.DataFrame(df_sreality)
    else:
        # or get newest sreality df, takes about 6 minutes
        from src.function_scripts import request_sreality_all
        df_sreality = request_sreality_all() 
        df_sreality.to_json("data/raw/df_sreality.json", orient="records")

    # get square meters (area) and flat type from name
    from src.function_scripts import name_to_area
    df_sreality['area'] = df_sreality.name.apply(name_to_area)
    df_sreality['flat_type'] = df_sreality.name.apply(lambda x: x.split()[2])

    # get link to listing and thumbnail image
    from src.function_scripts import get_link_and_image
    df_sreality = get_link_and_image(df_sreality)

    # clean sreality dataset
    df_sreality_clean = df_sreality[['locality', 'price', 'flat_type','area','gps','hash_id', 'url','image']].copy()

    # get latitude and logitude, manually adjusted based on results
    df_sreality_clean[['lat', 'lon']] = df_sreality_clean.gps.apply(lambda x: pd.Series({'lat': x['lat']+ 0.008, 'lon': x['lon']-0.008}))
    df_sreality_clean = df_sreality_clean.drop(columns = ["gps", "hash_id"])

    if update == False:
        # get bezrealitky df from last request
        df_bezrealitky = pd.read_json("data/raw/df_bezrealitky.json")
        df_bezrealitky = pd.DataFrame(df_bezrealitky)
    else:
        # or get newest bezrealitky df, takes about 10 minutes
        from src.function_scripts import request_bezrealitky
        bezrealitky_url = "https://www.bezrealitky.cz/vyhledat?estateType=BYT&location=exact&offerType=PRONAJEM&osm_value=%C4%8Cesko&regionOsmIds=R51684&currency=CZK"
        df_bezrealitky = request_bezrealitky(bezrealitky_url, 171)
        df_bezrealitky.to_json("data/raw/df_bezrealitky.json", orient="records")

    # convert flat_type to standard nomenclature
    mapping = {
        'DISP_1_KK': '1+kk',
        'DISP_2_KK': '2+kk',
        'DISP_3_KK': '3+kk',
        'DISP_4_KK': '4+kk',
        'DISP_1_1': '1+1',
        'DISP_2_1': '2+1',
        'DISP_3_1': '3+1',
        'DISP_4_1': '4+1',
        'DISP_5_1': '5+1',
        'DISP_7_1': '7+1',
        'GARSONIERA': '1+kk',
        'OSTATNI': 'atypické',
        'UNDEFINED': 'atypické',
    }

    df_bezrealitky['flat_type'] = df_bezrealitky['flat_type'].map(mapping)
    df_bezrealitky_clean = df_bezrealitky

    # remove non-czech properties
    df_sreality_clean = df_sreality_clean[
        (df_sreality_clean['lat'] >= 48.5) &
        (df_sreality_clean['lat'] <= 51.1) &
        (df_sreality_clean['lon'] >= 12.0) &
        (df_sreality_clean['lon'] <= 18.9)
    ]

    df_bezrealitky_clean = df_bezrealitky_clean[
        (df_bezrealitky_clean['lat'] >= 48.5) &
        (df_bezrealitky_clean['lat'] <= 51.1) &
        (df_bezrealitky_clean['lon'] >= 12.0) &
        (df_bezrealitky_clean['lon'] <= 18.9)
    ]

    # concatenate datasets
    df_all = pd.concat([df_sreality_clean, df_bezrealitky_clean], ignore_index=True)


    # change non-standard flat types to "other"
    standard = {
    '1+kk', '1+1',
    '2+kk', '2+1',
    '3+kk', '3+1',
    '4+kk', '4+1',
    '5+kk', '5+1',
    'atypické'
    }

    df_all['flat_type'] = df_all['flat_type'].where(
        df_all['flat_type'].isin(standard), other='other'
    )

    # save df_all as csv
    df_all.to_csv("data/processed/df_all.csv")

    # datasets for map
    df_heatmap = df_all[['lat', 'lon', 'price', 'area', 'flat_type']].copy()
    df_sreality_property = df_sreality_clean[['lat', 'lon', 'price','locality', 'flat_type', 'area', 'url', 'image']].copy()
    df_bezrealitky_property = df_bezrealitky_clean[['lat', 'lon', 'price','locality', 'flat_type', 'area', 'url', 'image']].copy()

    # save map datasets 
    df_heatmap.to_parquet("data/processed/df_heatmap.parquet")
    df_sreality_property.to_parquet("data/processed/df_sreality_property.parquet")
    df_bezrealitky_property.to_parquet("data/processed/df_bezrealitky_property.parquet")



    # Dataframe for analysis
    df_reg = df_all[['price', 'area', 'flat_type', 'locality', 'lat', 'lon']].dropna().copy()

    # Add distance to Prague as a variable
    prague_coords = (50.0873, 14.420109) # Old Town Square coords

    # Calculate distance using geopy
    df_reg['distance_prague_km'] = df_reg.apply(
        lambda row: geodesic((row['lat'], row['lon']), prague_coords).km,
        axis=1
    )

    # Drop lat/lon — not needed in regression
    df_reg = df_reg.drop(columns=['lat', 'lon'])

    # Extract district from locality
    df_reg['district'] = df_reg['locality'].str.extract(r'^([^,]+)')

    # Extract city from district (or locality)
    from src.function_scripts import extract_city
    df_reg['city'] = df_reg.apply(lambda row: extract_city(row['district'], row['locality']), axis=1)

    # Drop extreme outliers (top/bottom 1%)
    q_low  = df_reg['price'].quantile(0.01)
    q_high = df_reg['price'].quantile(0.99)
    df_reg = df_reg[(df_reg['price'] > q_low) & (df_reg['price'] < q_high)]

    # Keep only districts with enough observations
    min_obs = 10
    district_counts = df_reg['district'].value_counts()
    df_reg = df_reg[df_reg['district'].isin(district_counts[district_counts >= min_obs].index)]




    # Save analysis dataframe
    df_reg.to_csv("data/processed/df_reg.csv")

