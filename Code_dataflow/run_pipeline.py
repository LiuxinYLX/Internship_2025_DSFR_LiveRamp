# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-06-18


import datetime
import textwrap
import apache_beam as beam
from google.cloud import bigquery
from config.configuration import DatasetConfig
from apache_beam.options.pipeline_options import PipelineOptions

from pipeline.transforms_sql import (
    generate_clean_query
)
from config.utils import (
    load_yaml
)
from pipeline.transforms_beam import (
    InitializeErrorReason,
    stringify_error_reason,
    CheckDuplicateRows,
    CheckDuplicateKeys,
    CheckBarcodeLength,
    CheckDateFormat,
    StandardizeByFrequence,
    drop_error_reason,
    handle_missing_brand,
    complete_brand
)

from config.utils import do_query_job
import logging
logging.getLogger("apache_beam").setLevel(logging.ERROR)

def bq_schema_to_str(schema, extra=None):
    s = [f"{f.name}:{f.field_type.lower()}" for f in schema]
    if extra:
        s += [f"{name}:{ftype}" for name, ftype in extra]
    return ",".join(s)

def run():
    cfg = DatasetConfig()
    client = bigquery.Client()
    yaml_path = "config/dataset.yaml"
    raw_table = f"{cfg.project}.{cfg.dataset}.{cfg.raw_table}"
    clean_table = f"{cfg.project}.{cfg.dataset}.{cfg.clean_table}"
    excluded_table = f"{cfg.project}.{cfg.dataset}.{cfg.excluded_table}"

    schema = client.get_table(raw_table).schema
    all_columns = [col.name for col in schema] # List[str]
    std_cols, rules = load_yaml(cfg.dataset_type, yaml_path)

    options = PipelineOptions(
        runner="DataflowRunner", ### "DirectRunner"
        project=cfg.project,
        temp_location=f"gs://liuxin_workspace/staging",
        region="europe-west1",
        job_name=f"qa-{cfg.dataset_type}-{datetime.datetime.now().strftime('%Y-%m-%d-%Hh%Mm')}".lower(),
        service_account_email="premium-publisher-transfer@ds-fra-eu-non-pii-prod.iam.gserviceaccount.com",
        num_workers=2,
        max_num_workers=5,
        machine_type="e2-standard-2",
        network="ds-fra-eu-non-pii-prod",
        subnetwork="https://www.googleapis.com/compute/v1/projects/ds-fra-eu-non-pii-prod/regions/europe-west1/subnetworks/ds-fra-eu-non-pii-prod-01",
        save_main_session=True,
        no_use_public_ips=True,
        experiments=["enable_data_sampling"], ### Enable data sampling
        setup_file="./setup.py",
    )

    print("Truncate excluded and clean tables before writing...")
    client.query(f"TRUNCATE TABLE `{excluded_table}`").result()
    client.query(f"TRUNCATE TABLE `{clean_table}`").result()

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
    clean_query = generate_clean_query(cfg, client, "raw")
    do_query_job(cfg, client, "clean", None, clean_query)
    
    ################################################################################################
    ###                      Step1. Running Semantic-level Cleaning...                           ###
    ################################################################################################
    
    print(textwrap.dedent("""
            Running Semantic-level Cleaning...
                6. Handle missing brand names
                7. Complete brands
                8. Unify brand names\n"""))
    
    with beam.Pipeline(options=options) as pcoll:
        data = (
            pcoll
            | "Read Cleaned Dataset from BigQuery" >> beam.io.ReadFromBigQuery(table = clean_table)
            | "Handle missing local brand names" >> beam.ParDo(handle_missing_brand("local_brand_name"))
            | "Handle missing global brand names" >> beam.ParDo(handle_missing_brand("global_brand_name"))
            # | "Complete brands" >> beam.ParDo(complete_brand())
            | "Standardize Contents of Columns" >> StandardizeByFrequence(std_cols,all_columns)
        )

    ################################################################################################
    ###                                Step2. Data validation...                                 ###
    ################################################################################################
        print("Step 2: Data validation...")
    
    # with beam.Pipeline(options=options) as pcoll:
        data = data | "Initialize Error Reason Column" >> beam.ParDo(InitializeErrorReason())

        if "duplicate_row" in rules:
            data = data | "Check Duplicate Rows" >> CheckDuplicateRows(all_columns)
        if "duplicate_key" in rules:
            data = data | "Check duplicate keys" >> CheckDuplicateKeys(cfg.primary_keys)
        if "barcode_length" in rules:
            data = data | "Check barcode length" >> beam.ParDo(CheckBarcodeLength(cfg.barcode_columns))
        if "date_format" in rules:
            data = data | "Check date format" >> beam.ParDo(CheckDateFormat(cfg.date_columns))
        
        clean, excluded = data | "Split data into clean and excluded" >> beam.Partition(
            lambda x, _: 0 if not x.get("error_reason",[]) else 1, 2
        )

        clean_schema_str = bq_schema_to_str(schema)
        excluded_schema_str = bq_schema_to_str(schema, [("error_reason", "string")])

        print("Update Clean table...")
        clean = clean | "Drop error_reason from clean" >> beam.Map(drop_error_reason)
        
        clean = clean | "Write clean data to BigQuery" >> beam.io.WriteToBigQuery(
            clean_table,
            #schema="SCHEMA_AUTODETECT",
            schema=clean_schema_str,
            write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
        )

        print("Create Excluded table...")
        excluded = excluded | "Excluded : Convert error_reason list to string" >> beam.Map(stringify_error_reason)
        excluded = excluded | "Write excluded data to BigQuery" >> beam.io.WriteToBigQuery(
                excluded_table,
                #schema="SCHEMA_AUTODETECT",
                schema = excluded_schema_str,
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
            )
        


if __name__ == "__main__":
    run()