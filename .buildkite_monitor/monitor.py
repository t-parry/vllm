import pandas as pd
from dotenv import load_dotenv

import os
import json
import requests
from datetime import datetime
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

warnings.filterwarnings('ignore')


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# Load environment variables from .env file
load_dotenv()

BUILDKITE_API_TOKEN = os.getenv('BUILDKITE_API_TOKEN')
ORGANIZATION_SLUG='vllm'
PIPELINE_SLUG = 'ci-aws'
TODAY = datetime.utcnow().strftime('%Y-%m-%dT00:00:00Z') # it is UTC, so -2 hours from Finnish local time
WAITING_TIME_ALERT_THR = 14400 # 4 hours


# Set the URL
url = f"https://api.buildkite.com/v2/organizations/{ORGANIZATION_SLUG}/pipelines/{PIPELINE_SLUG}/builds"

# Set the query parameters
params = {
    'created_from': TODAY,
}

# Set the headers
headers = {
    'Authorization': f'Bearer {BUILDKITE_API_TOKEN}'
}

# Make the GET request
response = requests.get(url, headers=headers, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    df = pd.json_normalize(data)
else:
    print(f"Request failed with status code {response.status_code}")


useful_columns = ['id', 'web_url', 'url', 'number', 'state', 'cancel_reason', 'blocked', 'blocked_state', 'jobs']
d = df[useful_columns]

jobs_df = pd.json_normalize(df['jobs'].explode())
jobs_useful_columns = ['id', 'name', 'state', 'build_url', 'web_url', 'soft_failed', 'created_at', 'scheduled_at', 'runnable_at', 'started_at',	'finished_at', 'expired_at', 'retried', 'agent.id', 'agent.name', 'agent.web_url', 'agent.connection_state', 'agent.meta_data']
j = jobs_df[jobs_useful_columns]

result_df = d.drop(columns=['jobs']).merge(j.reset_index(drop=True), left_on='url', right_on='build_url', suffixes=['_build', '_job'], how='outer')

# filter to AMD tests:
result_df_amd = result_df[(result_df.name.notna()) & (result_df.name.str.contains('AMD'))]

result_df_amd['number'] = result_df_amd['number'].astype('int')


def calculate_wait_time(df):
    now_utc = pd.Timestamp.now(tz='UTC')
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    df['runnable_at'] = pd.to_datetime(df['runnable_at'], errors='coerce')

    # Calculate the difference in seconds
    df['waited_seconds'] = df.apply(
        lambda row: (row['started_at'] - row['runnable_at']).total_seconds() if pd.notna(row['started_at']) and pd.notna(row['runnable_at']) \
              else (now_utc - row['runnable_at']).total_seconds() if pd.notna(row['started_at']) and pd.isna(row['runnable_at']) \
                else None,
        axis=1
    )
    return df

result_df_amd = calculate_wait_time(result_df_amd)

def alert(df, wait_time_thr=WAITING_TIME_ALERT_THR):
    alerts = []
    for index, row in result_df_amd.iterrows():
        if row['waited_seconds'] > wait_time_thr:
            alert_message = f"Job {row['name']} from build number {row['number']} waited for {row['waited_seconds']} seconds (more than {wait_time_thr} or {wait_time_thr/3600} hours). More info at {row['web_url_job']}"
            alerts.append(alert_message)
        if (row['state_job'] == 'failed') & (row['soft_failed'] == False): 
            alert_message = f"Job {row['name']} from build number {row['number']} failed. More info at {row['web_url_job']}"
            alerts.append(alert_message)


    return alerts

alerts = alert(result_df_amd)


def send_email(alerts):
    # Both email and smtplib are standard parts of python and therefore can be imported without pip install
    # Sends email using gmail's username and app password that are in credentials.txt
    # Emails don't seem to go through to AMD email addresses, hence the @silo.ai

    credentials = open('credentials.txt').read().split(':')
    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    s.login(credentials[0], credentials[1])

    msg = MIMEMultipart()
    msg['Subject'] = "Test email"
    msg['From'] = credentials[0]
    msg['To'] = "hissu.hyvarinen@silo.ai"

    alerts_str = '\n'.join(alerts)
    msg.attach(MIMEText(alerts_str, 'plain'))

    s.send_message(msg)
    s.quit()

if alerts:
    print('sending email')
    send_email(alerts)  
else:
    print('No alerts this time')      