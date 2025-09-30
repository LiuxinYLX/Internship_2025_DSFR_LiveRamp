# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-05-13

import time
import pandas as pd
from config.configuration import DatasetConfig
from google.cloud import bigquery


def obtain_table_name(cfg:DatasetConfig, tableType: str) -> str:
    
    if tableType == "raw":
        table = cfg.raw_table
    elif tableType == "clean":
        table = cfg.clean_table
    elif tableType == "excluded":
        table = cfg.excluded_table
    else:
        raise ValueError("Invalid table type. Choose from 'raw', 'clean', or 'excluded'.")
    return table

def obtain_dataframe(cfg:DatasetConfig, client:bigquery.Client, tableName: str) -> pd.DataFrame:

    data = client.query(f"""
                        SELECT * FROM `{cfg.project}.{cfg.dataset}.{tableName}`
                        """).to_dataframe()
    
    return data

def load_check_rules(dataset_type: str, yaml_path: str = "config/check_rules.yaml") -> list:
    """
    Load check rules from a YAML file based on the dataset type.
    
    Args:
        dataset_type (str): The type of dataset (e.g., 'raw', 'clean').
        yaml_path (str): Path to the YAML file containing check rules.
        
    """
    import yaml
    with open(yaml_path, 'r') as file:
        rules = yaml.safe_load(file)
    return rules.get(dataset_type, {}).get("rules", [])


def time_it(func_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func_name} completed in {end - start:.2f} seconds.")
            return result
        return wrapper
    return decorator


def do_query_job(cfg:DatasetConfig,client:bigquery.Client, tableType, tableName, query:str) -> None:
    """
    Execute the query and save the result in the specified table
    """
    if tableType:
        table = obtain_table_name(cfg, tableType)
    else:
        table = tableName
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            destination = f"{cfg.project}.{cfg.dataset}.{table}",
            write_disposition = "WRITE_TRUNCATE",
        )
    )
    job.result()

def do_data2table_job(cfg: DatasetConfig, client:bigquery.Client, tableType, tableName:str, df:pd.DataFrame, schema:list) -> None:
    """
    Convert a dataframe to a table
    """
    if tableType:
        table = obtain_table_name(cfg, tableType)
    else:
        table = tableName
        
    job = client.load_table_from_dataframe(
        df,
        f"{cfg.project}.{cfg.dataset}.{table}",
        job_config=bigquery.LoadJobConfig(
            write_disposition = "WRITE_TRUNCATE",
            schema = schema
        )
    )
    job.result()