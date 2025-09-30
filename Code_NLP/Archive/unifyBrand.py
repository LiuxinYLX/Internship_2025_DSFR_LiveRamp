# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-07-21

import re
import tqdm
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

def normalize_alphanum(val: str) -> str:
    """
    Extract the root the value by removing non-alphanumeric characters.
    """
    if pd.isnull(val):
        return None
    return re.sub(r'[^A-Z0-9]', '', val).upper()



def generate_mapping_by_root(df: pd.DataFrame, col: str) -> dict:
    """
    L'OREAL & L OREAL
    L OREAL -> L'OREAL
    """
    df["__root"] = df[col].apply(normalize_alphanum)
    
    most_frequent = (
        df.groupby(["__root",col]).size()                             # Group by (__root, original, size)
        .reset_index(name="count")                                    # Group by (__root, original, count)
        .sort_values(["__root", "count"], ascending=[True, False])    # Order by (__root ASC, count DESC)
        .drop_duplicates(subset=["__root"])                           # Keep the most frequent original for each __root
        .set_index("__root")[col]                                     # Index: __root, Value: original
        .to_dict()                                                    # Dict { __root: original }
        )
    
    df[col] = df["__root"].map(most_frequent)

    mapping = {}
    for _,row in tqdm(df.iterrows(), desc=f"Mapping {col} by root"):
        original = row[col]
        rooted = row["__root"]
        if pd.isnull(original) or pd.isnull(rooted):
            continue
        mapping[original] =  most_frequent.get(rooted, original)

    # 这种写法不会修改传入的原始 DataFrame df
    # df = df.drop(columns=["__root"])
    # 这种写法可以修改传入的原始 DataFrame df
    df.drop(columns=["__root"], inplace=True)

    return mapping
