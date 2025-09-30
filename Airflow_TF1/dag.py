from datetime import datetime, timedelta
import airflow

from airflow.operators.bash import BashOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
# airflow 2.0+ uses:
# from airflow.models import BashOperator
# from airflow.providers.google.cloud.operators.bigquery import BigQueryOperator

default_args = {
    'owner': 'Liuxin',
    'depends_on_past': False,
    'email': ['liuxin.yang@liveramp.com'],
    'email_on_failure': False, # Do not send email
    'email_on_retry': False,
    'start_date': datetime(2025, 8, 12),
    'retries': 0,
    'retry_delay': timedelta(minutes=15),
}

with airflow.DAG('TF1_Template_hybride', 
                 default_args=default_args,
                 # schedule_interval="*/30 * * * *", 
                 schedule_interval=None,
                 catchup=False,
                 tags=["LCR","TF1","Template_hybride"]) as dag:
    executer = BashOperator(task_id="executer",
                            bash_command="python /home/airflow/gcs/dags/Airflow_TF1/main.py")