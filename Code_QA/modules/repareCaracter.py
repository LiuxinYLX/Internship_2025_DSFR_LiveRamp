# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-07-22

import pandas as pd
from correction.correction_dict import correction_dict_es_cinema

def repare_point_interrogation(schema:list, df:pd.DataFrame) -> pd.DataFrame:
    
    cols = [field.name for field in schema]

    def fix_text_if_needed(text):
        if text is not None and "?" in text:
            return correction_dict_es_cinema.get(text, text)
        return text

    for col in cols:
        df[col] = df[col].apply(lambda x: fix_text_if_needed(x)) 
    
    return df
