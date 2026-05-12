import requests
import pandas as pd
import time
import random
import math
import unidecode
import re


def request_sreality(page, category_main_str, category_type_str, locality_region_id=10):
    """
    Request data from sreality.cz API
    :param page: page number
    :param category_main_str: category of the property
    :param category_type_str: type of the offer
    :param locality_region_id: region id
    :return tuple: (json response or None, bool next page exists)
    """
    template_url = 'https://www.sreality.cz/api/cs/v2/estates?category_main_cb={category_main}&category_type_cb={category_type}&locality_region_id={locality_region_id}&per_page=60&page={page}'
    category_main_cb = {'flat':1, 'house':2, 'land':3 }
    category_type_cb = {'sell':1,'rent':2}
    url_path = template_url.format(category_main=category_main_cb[category_main_str],
                                   category_type=category_type_cb[category_type_str],
                                   locality_region_id=locality_region_id,
                                   page=page)
    time.sleep(random.uniform(0.5, 1.2))
    try:
        r = requests.get(url_path)
        r.raise_for_status()
        data = r.json()

    except Exception as e:
        print(e)
        return None, 0

    # total number of pages
    result_size = data.get('result_size', 0)
    n_pages = math.ceil(result_size / 60)

    return data, n_pages


def convert_sreality_data_to_df(sreality_data):
    if not isinstance(sreality_data, dict):
        return pd.DataFrame()
    if '_embedded' not in sreality_data or 'estates' not in sreality_data['_embedded']:
        return pd.DataFrame()
    data = sreality_data['_embedded']['estates']
    return pd.DataFrame(data)


def request_multiple_sreality(category_main_str, category_type_str, locality_region_id=10):
    list_of_dfs = []
    data, n_pages = request_sreality(
        1,
        category_main_str,
        category_type_str,
        locality_region_id
    )

    print(f"Region {locality_region_id}: {n_pages} pages")

    list_of_dfs.append(convert_sreality_data_to_df(data))

    # remaining pages
    for page in range(2, n_pages + 1):

        data, _ = request_sreality(
            page,
            category_main_str,
            category_type_str,
            locality_region_id
        )

        df = convert_sreality_data_to_df(data)

        if not df.empty:
            list_of_dfs.append(df)

    return pd.concat(list_of_dfs, ignore_index=True)


def request_sreality_all():
    list_of_dfs = []
    for region_id in range(1, 15):
        df = request_multiple_sreality('flat', 'rent', locality_region_id=region_id)
        if not df.empty:
            list_of_dfs.append(df)
    return pd.concat(list_of_dfs, ignore_index=True) if list_of_dfs else pd.DataFrame()


def get_link_and_image(df):
    base = "https://www.sreality.cz/detail"

    def slugify(text):
        return unidecode.unidecode(text).lower().replace(" ", "-")
    def clean_flat_type(x):
        if isinstance(x, str):
            return x.replace(" ", "")
        return ""

    def build_url(row):
        seo = row["seo"]
        type = clean_flat_type(row["flat_type"])
        locality = slugify(seo["locality"])
        hid = row["hash_id"]
        return f"{base}/pronajem/byt/{type}/{locality}/{hid}"

    df["url"] = df.apply(build_url, axis=1)

    df["image"] = df["_links"].apply(
        lambda x: x["images"][0]["href"] if "images" in x and len(x["images"]) > 0 else None)
    return df

def name_to_area(nm):
    splitted_str = nm.split()
    m2_idx = splitted_str.index('m²')
    return int(splitted_str[m2_idx - 1])
    
