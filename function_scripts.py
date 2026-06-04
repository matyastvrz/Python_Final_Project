import requests
import pandas as pd
import time
import random
import math
import unidecode
import re
import json

#---------------------
# sreality scraping
#---------------------

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
    



#---------------------
# bezrealitky scraping
#---------------------

def request_bezrealitky(search_url, max_pages=20):
    all_extracted_data = []
    session = requests.Session()
    
    # headers
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9"
    })

    for page_num in range(1, max_pages + 1):
        # build url for page
        current_url = f"{search_url}&page={page_num}"
        print(f"Fetching: Page {page_num}...")
        
        try:
            response = session.get(current_url, timeout=15)
            response.raise_for_status()
            
            # extract json from the __NEXT_DATA__ script tag
            html_content = response.text
            start_marker = '<script id="__NEXT_DATA__" type="application/json">'
            end_marker = '</script>'
            
            start_index = html_content.find(start_marker)
            if start_index == -1:
                print("End of results reached or structure changed.")
                break
                
            json_start = start_index + len(start_marker)
            json_end = html_content.find(end_marker, json_start)
            json_str = html_content[json_start:json_end]
            
            data = json.loads(json_str)
            apollo_cache = data.get('props', {}).get('pageProps', {}).get('apolloCache', {})
            
            # count listings
            listings_on_page = 0
            
            for key, value in apollo_cache.items():
                if key.startswith('Advert:'):
                    listings_on_page += 1
                    
                    # extract variable info
                    gps = value.get('gps', {})
                    main_img = value.get('mainImage', {})
                    img_id = main_img.get('__ref') if isinstance(main_img, dict) else None
                    
                    # get image url if exists
                    image_url = 'N/A'
                    if img_id and img_id in apollo_cache:
                        image_url = apollo_cache[img_id].get('url({"filter":"RECORD_MAIN"})', 'N/A')

                    all_extracted_data.append({
                        'locality': value.get('address({"locale":"CS"})', 'N/A'),
                        'price': value.get('price'),
                        'flat_type': value.get('disposition', 'N/A'),
                        'area': value.get('surface'),
                        'url': f"https://www.bezrealitky.cz/nemovitosti-byty-domy/{value.get('uri', '')}",
                        'image': image_url,
                        'lat': gps.get('lat') if isinstance(gps, dict) else None,
                        'lon': gps.get('lng') if isinstance(gps, dict) else None
                    })
            
            print(f"Found {listings_on_page} listings on page {page_num}.")
            
            # if no listings, end
            if listings_on_page == 0:
                print("No more listings found. Finishing.")
                break
            
            # wait between requests
            time.sleep(2.5)

        except Exception as e:
            print(f"Error on page {page_num}: {e}")
            break

    # convert to df
    df = pd.DataFrame(all_extracted_data)
    
    return df


def extract_city(district):
    """Map a district string to its parent city."""
    if not isinstance(district, str):
        return None
    d = district.strip()

    if d.startswith('Praha'):
        return 'Praha'
    if d.startswith('Brno'):
        return 'Brno'
    if d.startswith('Ostrava'):
        return 'Ostrava'
    if d.startswith('Plzeň') or d.startswith('Plzen'):
        return 'Plzeň'
    if d.startswith('Liberec'):
        return 'Liberec'
    if d.startswith('Olomouc'):
        return 'Olomouc'
    if d.startswith('České Budějovice'):
        return 'České Budějovice'
    if d.startswith('Hradec Králové'):
        return 'Hradec Králové'
    if d.startswith('Pardubice'):
        return 'Pardubice'
    if d.startswith('Zlín'):
        return 'Zlín'
    if d.startswith('Havířov'):
        return 'Havířov'
    if d.startswith('Kladno'):
        return 'Kladno'

    # for all other districts, use the district name as city
    return d
