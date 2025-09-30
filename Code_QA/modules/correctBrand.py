# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-07-21

import re
import tqdm
import time
import Levenshtein
import pandas as pd

from tqdm import tqdm
from google.cloud import bigquery
from collections import defaultdict
from config.configuration import DatasetConfig
from config.utils import (
    obtain_dataframe,
    do_data2table_job
)
tqdm.pandas()

################################################################################################
###                                      Constances                                          ###
################################################################################################


MISSING_BRANDS = [
    "AUTRES MARQUES", "OTHER BRANDS", "AUTRES", 
    "SANS MARQUE", "SANS MARQUES", "SANS MARQUES.", "SANS MARQUE.", "REFS.SANS MARQUE", "SANS MARQUE CORA",
    "CET", "MN", "\"1\"",
    "NO NAME", "PP NO NAME", "UNKNOWN", "?",
    "SANS", "SANS.", "SAN", 
]

REMOVE_TOKENS = [
    "PEM", "LOCAL", "PERM", "LIGNE DE","NUIT", "RIST", "S/VET", "MN", "PROMOCASH", "IMPORT", "NET", "LS", "NO NAME"
]

VERIFIED_BRANDS = [
    "PETIT BATEAU"
]

schema = [
            bigquery.SchemaField("brand", "STRING"),
            bigquery.SchemaField("standard_brand", "STRING")
        ]

################################################################################################
###                               5. Handle missing brand                                    ###
################################################################################################

def handle_missing_brand(df: pd.DataFrame, cols: list):
    start = time.time()
    for col in cols:
        mask = df[col].isin(MISSING_BRANDS)
        df.loc[mask, col] = None
    end = time.time()
    print(f"Handle missing brands completed in {end - start:.2f} seconds.")
    return df

def test_handle_missing_brand(df: pd.DataFrame, cols: list):
    for col in cols:
        assert df[col].isin(MISSING_BRANDS).sum() == 0, f"Column {col} still contains missing brands."

################################################################################################
###                                    6. Complete brand                                     ###
################################################################################################

def complete_brand(tab1, tab2: pd.DataFrame) -> pd.DataFrame:
    """
    Complete the brand names in tab1 using the brand names from tab2.
    
    Args:
        tab1: DataFrame with incomplete brand names.
        tab2: DataFrame OpenEAN.
    
    """
    start = time.time()
    tab2_dedup = tab2.drop_duplicates(subset=["code"])

    tab_merged = tab1.merge(
        tab2_dedup[["code", "brand"]],
        left_on="barcode",
        right_on="code",
        how="left"
    )

    tab_merged["local_brand_name"] = tab_merged["local_brand_name"].fillna(tab_merged["brand"])
    tab_merged["global_brand_name"] = tab_merged["global_brand_name"].fillna(tab_merged["brand"])

    tab_merged = tab_merged.drop(columns=["brand", "code"])

    if len(tab1) != len(tab_merged):
        raise ValueError("The number of rows in the merged DataFrame does not match the original DataFrame.")
    
    end = time.time()
    print(f"Complete brand names completed in {end - start:.2f} seconds.")
    return tab_merged

################################################################################################
###                                  7. Unify brand name                                     ###
################################################################################################

def choose_most_frequent(frequency: dict, val1: str, val2: str) -> str:
    """
    Choose the most frequent brand name between b1 and b2 based on frequency dictionary.
    """
    if frequency.get(val1, 0) > frequency.get(val2, 0):
        return val1
    elif frequency.get(val1, 0) < frequency.get(val2, 0):
        return val2
    else:
        return max(val1,val2, key=lambda x: len(x))


def generate_mapping_ngram(brands: list, frequency: dict, n, min_share: int, threshold: float) -> dict:

    def get_ngrams(text: str, n: int) -> set:
        """
        Generate n-grams from the input text.
        """
        text = str(text).upper()
        return set([text[i:i+n] for i in range(len(text)-n+1)]) if len(text) >= n else set([text])

    
    mapping = {}
    checked = set()

    brands_ngrams = defaultdict(set)
    ngram_to_brands = defaultdict(set)
    
    for b in brands:
        grams = get_ngrams(b, n)
        brands_ngrams[b] = grams
        for gram in grams:
            ngram_to_brands[gram].add(b)
    
    for b in brands:
        candidates_count = defaultdict(int)

        grams = brands_ngrams[b]

        for gram in grams:
            for other in ngram_to_brands[gram]:
                if other == b:
                    continue

                pair = tuple(sorted([b, other]))
                if pair in checked:
                    continue
                
                candidates_count[other] += 1

        candidates = [other for other, count in candidates_count.items() if count >= min_share]

        if not candidates:
            continue

        for other in candidates:
            pair = tuple(sorted([b, other]))
            checked.add(pair)
            # score = difflib.SequenceMatcher(None, b, other).ratio()
            score = Levenshtein.ratio(b, other)
            if score >= threshold:
                std = choose_most_frequent(frequency, b, other)
                mapping[b] = std
                mapping[other] = std
                                               
    return mapping


def generate_mapping_by_similarity(brands: list, frequency: dict, threshold: float) -> dict:
    """
    LA COURONNE & LA COURONN 
    LA COURONN -> LA COURONNE
    """
    mapping_similarity = {}

    # Bucket brands by first letter
    letter_bucket = defaultdict(list)
    ### for brand in tqdm(brands,desc="Bucket brands by first letter"):
    for brand in brands:
        if not brand or not isinstance(brand, str):
            continue
        first_letter = brand[0].upper()
        letter_bucket[first_letter].append(brand)
    
    # Bucket brands by length
    ### for first_letter, bucket in tqdm(letter_bucket.items(), desc="Bucket brands by length"):
    for first_letter, bucket in letter_bucket.items():
        length_bucket = defaultdict(list)
        for brand in bucket:
            length_bucket[len(brand)].append(brand)
        all_lengths = sorted(length_bucket.keys())
        mapping = {}

        for i in range(min(all_lengths), max(all_lengths) - 1):
            sub_brands_bucket = length_bucket[i] + length_bucket[i + 1]
            mapping_ngram = generate_mapping_ngram(
                sub_brands_bucket, frequency, n=3, min_share=2, threshold=threshold
            )
            mapping.update(mapping_ngram)
        
        mapping_similarity.update(mapping)

    return mapping_similarity


def extract_root(val: str) -> str:
    if pd.isnull(val):
        return None
    
    upper = str(val).upper()

    for b in VERIFIED_BRANDS:
        if b in upper:
            return b

    parts = upper.split()  # ["PETIT", "BATEAU", "62", "NUIT", "RIST"]
    val_cleaned = [w for w in parts if w not in REMOVE_TOKENS and not w.isdigit()] # val_cleaned = ["PETIT", "BATEAU"]
    
    return ' '.join(val_cleaned) if val_cleaned else upper


def unify_brand(df: pd.DataFrame, cols:list, cfg:DatasetConfig, client: bigquery.Client, if_mapping:bool, mapping_name:str): 

    client = bigquery.Client()
    mapping_similarity = {}
    values_all = set()
    
    start = time.time()
    if if_mapping:

        df_mapping = obtain_dataframe(cfg, client, mapping_name)
        mapping_similarity = dict(zip(df_mapping["brand"], df_mapping["standard_brand"]))
        
        for col in cols:
            df[col] = df[col].apply(extract_root)
            df[col] = df[col].replace(mapping_similarity)
    else:
        ######################### Brand extraction #########################
        start = time.time()
        brand_counter = {}
        for col in cols:
            df[col] = df[col].apply(extract_root)
            values = df[col].dropna()

            for brand in values:
                brand_counter[brand] = brand_counter.get(brand, 0) + 1
            
            values_all.update(values.unique())
        end = time.time()
        
        print(f"""
        Unify N°1: Brand extraction completed in {end - start:.2f} seconds.""")
        ######################### Mapping by similarity #########################
        start = time.time()
        brands = sorted(list(values_all))
        mapping_similarity = generate_mapping_by_similarity(brands, brand_counter, threshold=0.85)
        for col in cols:
            df[col] = df[col].replace(mapping_similarity)
        end = time.time()
        
        print(f"""
        Unify N°2: Mapping by similarity created in {end - start:.2f} seconds.""")
        ######################### Save mapping #########################
        start = time.time()
        mapping_similarity_df = pd.DataFrame(
            [{"brand": k, "standard_brand": v} for k, v in mapping_similarity.items()]
        )
        mapping_similarity_df = mapping_similarity_df.drop_duplicates(subset=["brand", "standard_brand"])
        
        do_data2table_job(cfg, client, None,f"mapping_{cfg.dataset_type}", mapping_similarity_df, schema)
        end = time.time()
        
        print(f"""
        Unify N°3: Mapping saved in {end - start:.2f} seconds.""")
        
    end = time.time()
    print(f"Unify brand names completed in {end - start:.2f} seconds.\n")
    
    return df