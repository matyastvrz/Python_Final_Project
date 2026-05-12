import requests
import pandas as pd
import time
import random


def request_sreality(page, category_main_str, category_type_str, locality_region_id=10):
    """
    Request data from sreality.cz API
    :param page: page number
    :param category_main_str: category of the property
    :param category_type_str: type of the offer
    :param locality_region_id: region id
    :return json: json response
    """
    template_url = 'https://www.sreality.cz/api/cs/v2/estates?category_main_cb={category_main}&category_type_cb={category_type}&locality_region_id={locality_region_id}&per_page60&page={page}'
    category_main_cb = {'flat':1, 'house':2, 'land':3 }
    category_type_cb = {'sell':1,'rent':2}
    url_path = template_url.format(category_main=category_main_cb[category_main_str],
                                   category_type=category_type_cb[category_type_str],
                                   locality_region_id=locality_region_id,
                                   page=page)
    time.sleep(random.uniform(0.5, 1.2))
    try:
        r = requests.get(url_path)
    except Exception as e:
        print(e)
        return None
    return r.json()


def convert_sreality_data_to_df(sreality_data):
    data = sreality_data['_embedded']['estates']
    return pd.DataFrame(data)


def request_multiple_sreality(start_page, end_page, category_main_str, category_type_str, locality_region_id=10):
    visit_pages = range(start_page, end_page)
    list_of_dfs = []
    for page in visit_pages:
        df = convert_sreality_data_to_df(request_sreality(page, category_main_str, category_type_str, locality_region_id))
        list_of_dfs.append(df)
    return pd.concat(list_of_dfs)
