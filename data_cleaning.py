import pandas as pd
import json 
import os
import requests  
import time
import re 
import random 

def process_data(update = False):

    if update == False:
        # get sreality df from last request
        df_sreality = pd.read_json("data/df_sreality.json")
        df_sreality = pd.DataFrame(df_sreality)
    else:
        # or get newest sreality df, takes about 6 minutes
        from function_scripts import request_sreality_all
        df_sreality = request_sreality_all() 
        df_sreality.to_json("data/df_sreality.json", orient="records")

    # get square meters (area) and flat type from name
    from function_scripts import name_to_area
    df_sreality['area'] = df_sreality.name.apply(name_to_area)
    df_sreality['flat_type'] = df_sreality.name.apply(lambda x: x.split()[2])

    # get link to listing and thumbnail image
    from function_scripts import get_link_and_image
    df_sreality = get_link_and_image(df_sreality)

    # clean sreality dataset
    df_sreality_clean = df_sreality[['locality', 'price', 'flat_type','area','gps','hash_id', 'url','image']].copy()

    # get latitude and logitude, manually adjusted based on results
    df_sreality_clean[['lat', 'lon']] = df_sreality_clean.gps.apply(lambda x: pd.Series({'lat': x['lat']+ 0.008, 'lon': x['lon']-0.008}))
    df_sreality_clean = df_sreality_clean.drop(columns = ["gps", "hash_id"])

    if update == False:
        # get bezrealitky df from last request
        df_bezrealitky = pd.read_json("data/df_bezrealitky.json")
        df_bezrealitky = pd.DataFrame(df_bezrealitky)
    else:
        # or get newest bezrealitky df, takes about 10 minutes
        from function_scripts import request_bezrealitky
        bezrealitky_url = "https://www.bezrealitky.cz/vyhledat?estateType=BYT&location=exact&offerType=PRONAJEM&osm_value=%C4%8Cesko&regionOsmIds=R51684&currency=CZK"
        df_bezrealitky = request_bezrealitky(bezrealitky_url, 171)
        df_bezrealitky.to_json("data/df_bezrealitky.json", orient="records")

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

    df_all = pd.concat([df_sreality_clean, df_bezrealitky_clean], ignore_index=True)

    # remove non-czech properties (approximate)
    df_all = df_all[
        (df_all['lat'] >= 48.5) &
        (df_all['lat'] <= 51.1) &
        (df_all['lon'] >= 12.0) &
        (df_all['lon'] <= 18.9)
    ]

    # save df_all as csv
    df_all.to_csv("data/df_all.csv")

    # datasets for map
    df_heatmap = df_all[['lat', 'lon', 'price', 'area']].copy()
    df_sreality_property = df_sreality_clean[['lat', 'lon', 'price','locality', 'flat_type', 'area', 'url', 'image']].copy()
    df_bezrealitky_property = df_bezrealitky_clean[['lat', 'lon', 'price','locality', 'flat_type', 'area', 'url', 'image']].copy()

    # save map datasets 
    df_heatmap.to_parquet("data/df_heatmap.parquet")
    df_sreality_property.to_parquet("data/df_sreality_property.parquet")
    df_bezrealitky_property.to_parquet("data/df_bezrealitky_property.parquet")