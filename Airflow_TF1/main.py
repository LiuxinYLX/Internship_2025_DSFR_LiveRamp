#!/usr/bin/env python
# coding: utf-8


from LVR_API import (CleanRoom, create_table_parametre_question,
                     create_table_trigger_question, run_query_sql, envoi_email)
import config_email, config_projet
from google.cloud import bigquery
import datetime
import pandas as pd
import yaml


project_id = config_projet.project_id
data_set = config_projet.data_set
parametre_info_table = config_projet.parametre_info_table
trigger_info_table = config_projet.trigger_info_table
secret_project = config_projet.secret_project
secret_id = config_projet.secret_id

subject = config_email.subject
to_emails = config_email.to_emails
info_html = config_email.html_info

with open("/home/airflow/gcs/dags/Airflow_TF1/config_mapping_question.yaml","r") as yaml_file:
    mapping_tigger = yaml.safe_load(yaml_file)


id_parametre_info_table = ".".join([project_id, data_set, parametre_info_table])
id_trigger_info_table = ".".join([project_id, data_set, trigger_info_table])

for t_q in mapping_tigger["trigger_question"]:
    if t_q["trigger"] is not None:
        for j in t_q["trigger"].values():
            if j not in mapping_tigger["parameter_question"]["parameter"]:
                raise ValueError(f"Parameter {j} not found in trigger question {t_q['id']}")

client = bigquery.Client(project=project_id)

LVR = CleanRoom(secret_project=secret_project, secret_id=secret_id)

infos_parametre = LVR.get_question_runs(mapping_tigger["parameter_question"]["id"])

print(infos_parametre)

create_table_trigger_question(project_id, data_set, trigger_info_table)
create_table_parametre_question(project_id, data_set, parametre_info_table)

detected = []
QUERY = (
    f'SELECT distinct id FROM {id_parametre_info_table}'
)
query_job = client.query(QUERY)  # API request
rows = query_job.result()  # Waits for query to finish

for row in rows:
    print(row.id)
    detected.append(row.id)

time_str_now = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
for i in infos_parametre:
    error = False
    error_message = ""
    if i["id"] not in detected:
        print("id: {0}, name: {1}".format(i["id"], i["name"]))
        zz = pd.DataFrame([i])
        zz.drop(columns=["status", "completedAt",
                         'partitionParameters', 'runMetadata', 'failureReason',
                         'isRetryable', 'runMessage'], inplace=True)
        zz_html = zz.to_html(index=False)
        html_info = info_html.format(zz_html)
        # envoi_email(html_info, subject, to_emails)
        for mtp in mapping_tigger["parameter_question"]["parameter"]:
            if mtp not in i["parameters"].keys():
                print(f"Missing parameter: {mtp}")
                error = True
                error_message = f"Missing parameter: {mtp}"
        if error:
            QUERY = (
                f'INSERT INTO {id_parametre_info_table} (id,run_name,detect_time,parameters,raw,status,status_infos) VALUES ("{i["id"]}","{i["name"]}",CURRENT_TIMESTAMP(),"{str(i["parameters"])}","{str(i)}","error","{error_message}")'
            )
            run_query_sql(client, QUERY)
        else:

            QUERY = (

                f'INSERT INTO {id_parametre_info_table} (id,run_name,detect_time,parameters,raw,status,status_infos) VALUES ("{i["id"]}","{i["name"]}",CURRENT_TIMESTAMP(),"{str(i["parameters"])}","{str(i)}","detected","")'
            )
            run_query_sql(client, QUERY)
            for j in mapping_tigger["trigger_question"]:
                id_t = j["id"]
                if j["trigger"] is not None:
                    param = {k: i["parameters"][v] for k, v in j["trigger"].items()}
                else:
                    param = {}
                
                print(param)
                info_t_q = LVR.create_run(id_t, {"name": i["name"] + f"_{time_str_now}", "parameters": param})
                print(info_t_q)
                QUERY_2 = (
                    f'INSERT INTO {id_trigger_info_table} (id,trigger_time,trigger_job_id,trigger_job_name,parameters,raw,status) VALUES ("{i["id"]}",CURRENT_TIMESTAMP(),"{str(info_t_q["id"])}","{str(info_t_q["name"])}","{str(param)}","str({info_t_q})","trigger")'
                )
                run_query_sql(client, QUERY_2)