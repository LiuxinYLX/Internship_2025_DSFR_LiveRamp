import datetime
import os
import pytz
import requests, base64

from urllib3.exceptions import InsecureRequestWarning
import json
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
# from google.api_core.exceptions import NotFound

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from google.cloud import secretmanager


class BaseApi():
    def __init__(self, secret_project, secret_id):
        cle = json.loads(access_secret_version(secret_project, secret_id))
        self.org_uuid = cle["HABU_ORGANIZATION"]
        self.client = cle["HABU_API_CLIENT"]
        self.secret = cle["HABU_API_SECRET"]
        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
        self.api = os.getenv("HABU_API", "https://api.habu.com")
        self.access_token, self.expire_time = self.login()

    def login(self):
        client = self.client
        secret = self.secret

        if not client or not secret:
            raise Exception("API Client and Secret must be defined in your environment. Please check documentation!")

        usrPass = "%s:%s" % (client, secret)
        b64Val = base64.b64encode(usrPass.encode()).decode()

        data = {'grant_type': 'client_credentials'}
        response = requests.post("%s/v1/oauth/token" % self.api,
                                 headers={"Authorization": "Basic %s" % b64Val},
                                 data=data,
                                 verify=False)

        if response.status_code != 200:
            raise Exception("Unable to login to habu API. Check your environment and settings!")

        data = response.json()
        access_token = data["accessToken"]
        expires_at = data["expiresAt"]
        return access_token, datetime.datetime.strptime(expires_at, '%Y-%m-%dT%H:%M:%S.%f%z')

    def get_token(self):
        if self.expire_time < pytz.UTC.localize(datetime.datetime.now()):
            self.access_token, self.expire_time = self.login()
        return self.access_token

    def post(self, url, body=None):
        response = requests.post(url,
                                 verify=False,
                                 headers={'Authorization': 'Bearer %s' % self.get_token()},
                                 json=body)
        if response.status_code != 200:
            response.raise_for_status()
        return response.json()

    def get(self, url, key=None):
        response = requests.get(url,
                                verify=False,
                                headers={"Authorization": "Bearer %s" % self.get_token()})
        if response.status_code != 200:
            response.raise_for_status()

        if len(response.text) == 0:
            return []
        else:
            if key:
                return response.json()[key]
            else:
                return response.json()


class CleanRoom(BaseApi):
    def __init__(self, secret_project, secret_id):
        BaseApi.__init__(self, secret_project, secret_id)

    def get_clean_rooms(self):
        return self.get("%s/v1/cleanrooms" % self.api)

    def get_clean_room_questions(self, cleanroom_uuid):
        return self.get("%s/v1/cleanrooms/%s/cleanroom-questions"
                        % (self.api, cleanroom_uuid))

    def get_question_runs(self, question_uuid, limit=500, offset=0):
        return self.get("%s/v1/cleanroom-questions/%s/cleanroom-question-runs?limit=%s&offset=%s"
                        % (self.api, question_uuid, limit, offset))

    def get_question_run_data(self, run_uuid, limit=500, offset=0):
        return self.get("%s/v1/cleanroom-question-runs/%s/data?limit=%s&offset=%s"
                        % (self.api, run_uuid, limit, offset))

    def get_run_data_count(self, run_uuid):
        return self.get("%s/v1/cleanroom-question-runs/%s/data/count"
                        % (self.api, run_uuid))

    def create_run(self, question_uuid, create_run_parameters):
        return self.post("%s/v1/cleanroom-questions/%s/create-run"
                         % (self.api, question_uuid), create_run_parameters)


def access_secret_version(project_id, secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(name=name)
    payload = response.payload.data.decode("UTF-8")
    return payload


def create_table_parametre_question(project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id, location="EU")

    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("run_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("detect_time", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("parameters", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("raw", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("status_infos", "STRING", mode="REQUIRED")
    ]
    try:
        client.get_table(f"{project_id}.{dataset_id}.{table_id}")  # Make an API request.

    except NotFound:
        print("Table {} is not found.".format(f"{project_id}.{dataset_id}.{table_id}"))
        table = bigquery.Table(f"{project_id}.{dataset_id}.{table_id}", schema=schema)
        result = client.create_table(table)
        print(result)


def create_table_trigger_question(project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id, location="EU")

    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("trigger_time", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("trigger_job_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("trigger_job_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("parameters", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("raw", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
    ]
    try:
        client.get_table(f"{project_id}.{dataset_id}.{table_id}")  # Make an API request.

    except NotFound:
        print("Table {} is not found.".format(f"{project_id}.{dataset_id}.{table_id}"))
        table = bigquery.Table(f"{project_id}.{dataset_id}.{table_id}", schema=schema)
        result = client.create_table(table)
        print(result)


def run_query_sql(client, query_sql):
    print(client.query(query_sql).result())


def envoi_email(info_html, subject, to_emails):
    key = access_secret_version("ds-fra-eu-non-pii-prod", "SENDGRID_API_KEY", "latest")

    message = Mail(from_email='no-reply@liveramp.com',
                   to_emails=to_emails,
                   subject=subject,
                   html_content=info_html)

    try:
        sendgrid_client = SendGridAPIClient(
            key
        )
        response = sendgrid_client.send(message)
    except Exception as e:
        print(e.message)