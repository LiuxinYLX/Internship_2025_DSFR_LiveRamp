# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-06-02

import time
import pandas as pd
import textwrap
from google.cloud import bigquery
from config.configuration import DatasetConfig
from config.utils import (
    obtain_dataframe,
    do_query_job,
    do_data2table_job
)
from modules.correctBrand import (
    handle_missing_brand,
    complete_brand,
    unify_brand,
)

from modules.generateQuery import (
    generate_clean_query,
    generate_check_exclude_query,
    generate_warning_hierarchy

)



class DataCleaningPipeline:
    def __init__(self, config: DatasetConfig):
        self.cfg = config
        self.client = bigquery.Client(project=self.cfg.project)
        self.schema = self.client.get_table(f"{self.cfg.project}.{self.cfg.dataset}.{self.cfg.raw_table}").schema

    def run(self):
        ################################################################################################
        ###                       Step1. Running Format-level Cleaning...                            ###
        ################################################################################################
        print(textwrap.dedent("""
            Step 1: Data Cleaning...
            Running Format-level Cleaning...
                1. Normalise whitespace
                2. Convert to upper case
                3. Delete spaces before and after
                4. Remove diacritics (accents)
                5. Handling missing values\n"""))
        
        start = time.time()
        clean_query = generate_clean_query(self.cfg, self.client, "raw")
        do_query_job(self.cfg, self.client, "clean", None, clean_query)
        end = time.time()
        print(f"--> Format-level cleaning finished in {end - start:.2f} seconds.")

        ################################################################################################
        ###                      Step1. Running Semantic-level Cleaning...                           ###
        ################################################################################################
        print(textwrap.dedent("""
            Running Semantic-level Cleaning...
                6. Handle missing brand names
                7. Complete brands
                8. Unify brand names\n"""))
        # brands = [field.name for field in self.schema if "brand" in field.name.lower()]
        start = time.time()
        brands = ["local_brand_name", "global_brand_name"]
        
        open_ean_food = obtain_dataframe(self.cfg, self.client, "OpenEAN_Food")
        open_ean_petfood = obtain_dataframe(self.cfg, self.client, "OpenEAN_PetFood")
        open_ean_beauty = obtain_dataframe(self.cfg, self.client, "OpenEAN_Beauty")
        open_ean_products = obtain_dataframe(self.cfg, self.client, "OpenEAN_Products")
        open_ean_data = pd.concat([open_ean_food, open_ean_petfood, open_ean_beauty, open_ean_products], ignore_index=True)
    
    
        clean_data = (
            obtain_dataframe(self.cfg, self.client, self.cfg.clean_table)
            .pipe(handle_missing_brand, brands)
            .pipe(complete_brand, open_ean_data)
            #.pipe(unify_brand, brands, self.cfg, self.client, if_mapping = False, mapping_name = None)
            .pipe(unify_brand, brands, self.cfg, self.client, if_mapping = True, mapping_name = "mapping_supermarket")
        )

        do_data2table_job(self.cfg, self.client, "clean", None, clean_data, self.schema)
        end = time.time()
        print(f"--> Semantic-level cleaning finished in {end - start:.2f} seconds.\n")

        ################################################################################################
        ###                                Step2. Data validation...                                 ###
        ################################################################################################

        print("Step 2: Data validation...\n")
        exclude_query,_ = generate_check_exclude_query(self.cfg, self.client)
        do_query_job(self.cfg, self.client, "excluded", None, exclude_query)

        generate_warning_hierarchy(self.cfg, self.client)
        print("Pipeline finished.")


if __name__ == "__main__":    
    datacfg = DatasetConfig()
    pipeline = DataCleaningPipeline(datacfg)
    pipeline.run()
