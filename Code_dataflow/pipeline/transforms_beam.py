# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-06-18

import re
import datetime
import apache_beam as beam
from typing import Dict, List
from google.cloud import bigquery


################################################################################################
###                                     Data validation                                      ###
################################################################################################

class InitializeErrorReason(beam.DoFn):
    def process(self, element: Dict):
        element["error_reason"] = []
        yield element


class CheckDuplicateRows(beam.PTransform):
    def __init__(self, all_columns: List[str]):
        self.all_columns = all_columns

    def expand(self, pcoll):
        return (
            pcoll
            | "Key by all columns" >>  beam.Map(lambda x: (tuple(x[col] for col in self.all_columns),x))
            | "Group by all columns" >> beam.GroupByKey()
            | "Add error reason" >> beam.FlatMap(self._add_error_reason)
        )
        
    def _add_error_reason(self, key_value_pair):
        _, records = key_value_pair
        if len(records) > 1:
            for r in records:
                r["error_reason"].append("all_line_duplicate")
                yield r
        else:
            yield records[0]

class CheckDuplicateKeys(beam.PTransform):
    def __init__(self, primary_keys: List[str]):
        self.key_columns = primary_keys

    def expand(self, pcoll):
        return (
            pcoll
            | "Key by primary key" >> beam.Map(lambda x: (tuple(x[k] for k in self.key_columns), x))
            | "Group by primary key" >> beam.GroupByKey()
            | "Add error reason for duplicate keys" >> beam.FlatMap(self._add_error_reason)
        )
        
    def _add_error_reason(self, key_value_pair):
        _, records = key_value_pair
        if len(records) > 1:
            for r in records:
                r["error_reason"].append("primary_key_duplicate")
                yield r
        else:
            yield records[0]

class CheckBarcodeLength(beam.DoFn):
    def __init__(self, barcode_column: List[str]):
        self.barcode_column = barcode_column

    def process(self, element: Dict):
        for col in self.barcode_column:
            value = element.get(col)
            if value and len(str(value)) not in (7, 8, 10, 13):
                element["error_reason"].append("wrong_barcode_length")
        yield element
               

class CheckDateFormat(beam.DoFn):
    def __init__(self, date_columns: List[str]):
        self.date_columns = date_columns
        assert isinstance(self.date_columns, list), "date_columns must be a list of field names."


    def process(self, element: Dict):
        for col in self.date_columns:
            value = element.get(col)
            if value:
                try:
                    date = datetime.datetime.strptime(value, "%Y-%m-%d")
                    if date.year != 9999 and not (1963 <= date.year <= 2100):
                        element["error_reason"].append("wrong_date")
                except Exception:
                    element["error_reason"].append("wrong_date")
        yield element

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
###                                Handle missing brand names                                ###
################################################################################################

class handle_missing_brand(beam.DoFn):
    def __init__(self, col:str):
        # only parameters
        self.col = col 

    def process(self, element: Dict):
        value = element.get(self.col)
        if value in MISSING_BRANDS:
            element[self.col] = None
        yield element

 
################################################################################################
###                                    Complete brand                                     ###
################################################################################################

class complete_brand(beam.PTransform):
    def __init__(self, common_field_1 = "barcode", common_field_2 = "code", brand_field = "brand"):
        self.common_field_1 = common_field_1
        self.common_field_2 = common_field_2
        self.brand_field = brand_field

    def expand(self, inputs):
        tab1 = inputs["tab1"]
        tab2 = inputs["tab2"]
        
        left = tab1 | "L:KV(barcode,row)" >> beam.map(lambda r: (r.get(self.common_field_1),r))
        right = tab2 | "R:KV(code,brand)" >> beam.map(lambda r: (r.get(self.common_field_2),r.get(self.brand_field)))

        
        
        def attach_brand(input):
            _, groups = input
            tab1 = groups.get('L', [])
            tab2 = groups.get('R', [])

            brand = next((b for b in tab2 if b), None)
            for row in tab1:
                if brand:
                    if not row.get("local_brand_name"):
                        row["local_brand_name"] = brand
                    if not row.get("global_brand_name"):
                        row["global_brand_name"] = brand
                yield row

        return ({'L': left, 'R': right}
                  | "Join two tables on barcode=code" >> beam.CoGroupByKey()
                  | "Complete brand names" >> beam.FlatMap(attach_brand))

################################################################################################
###                                StandardizeByFrequence                                    ###
################################################################################################                       

def Nomalize(value: str) -> str:
    """Normalize a string by removing non-alphanumeric characters and converting to uppercase."""
    if value:
        return re.sub(r'[^A-Z0-9]', '', value.upper())
    return None

def stringify_error_reason(x):
    if isinstance(x.get("error_reason"), list):
        x["error_reason"] = ", ".join(x["error_reason"])
    return x

class AddNormalizedField(beam.DoFn):
    def __init__(self, col: str):
        self.col = col

    def process(self, element: Dict):
        value = element.get(self.col)
        if value and isinstance(value, str):
            element[f"_{self.col}_normalized"] = Nomalize(value)
        else:
            element[f"_{self.col}_normalized"] = None
        yield element

class ExtracOrigNormPair(beam.DoFn):
    def __init__(self, col: str):
        self.col = col

    def process(self, element: Dict):
        orig_value = element.get(self.col)
        norm_value = element.get(f"_{self.col}_normalized")
        if orig_value and norm_value:
            yield (norm_value, orig_value)
        else:
            yield (None, orig_value)


class CountMostFrequent(beam.CombineFn):
    def create_accumulator(self):
        return {}

    def add_input(self, accumulator, input):
        accumulator[input] = accumulator.get(input, 0) + 1
        return accumulator

    def merge_accumulators(self, accumulators):
        merged = {}
        for acc in accumulators:
            for k, v in acc.items():
                merged[k] = merged.get(k, 0) + v
        return merged

    def extract_output(self, accumulator):
        if not accumulator:
            return None
        return max(accumulator.items(), key=lambda x: x[1], default=(None, None))[0]
    
class ApplyMapping(beam.DoFn):
    def __init__(self, col):
        self.col = col

    def process(self, element: Dict, mapping: Dict[str, str]):
        norm = element.get(f"_{self.col}_normalized")
        if mapping and norm in mapping:
            element[self.col] = mapping[norm]
        yield element



class StandardizeByFrequence(beam.PTransform):
    def __init__(self, columns: List[str], output_columns: List[str]):
        self.columns = columns
        self.output_columns = output_columns

    def expand(self, pcoll):
        for col in self.columns:
            pcoll = pcoll | f"Add Normalized_{col}" >> beam.ParDo(AddNormalizedField(col))
            
            most_frequent = (
                pcoll
                | f"Extract Pairs (Original, Normalized): {col}" >> beam.ParDo(ExtracOrigNormPair(col))
                | f"Group By Normalized: {col}" >> beam.GroupByKey()
                | f"Find the Most Frequent Form: {col}" >> beam.Map(lambda kv: (kv[0], max(set(kv[1]), key=kv[1].count)))
            )   
            mapping = beam.pvalue.AsDict(most_frequent)

            pcoll = pcoll | f"Apply Mapping to Original Column: {col}" >> beam.ParDo(ApplyMapping(col), mapping)

        pcoll = pcoll | f"Remove all Temp Fields" >> beam.Map(lambda x: {k: v for k, v in x.items() if k in self.output_columns})
            
        return pcoll
    

def drop_error_reason(x):
    if "error_reason" in x:
        x = {k: v for k, v in x.items() if k != "error_reason"}
    return x