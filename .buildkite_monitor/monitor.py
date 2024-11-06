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
GMAIL_USERNAME = os.getenv('GMAIL_USERNAME')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
ORGANIZATION_SLUG='vllm'
PIPELINE_SLUG = 'ci-aws'
TODAY = (datetime.utcnow() - pd.Timedelta(days=1)).strftime('%Y-%m-%dT22:00:00Z') # it is UTC, so -2 hours from Finnish local time
WAITING_TIME_ALERT_THR = 14400 # 4 hours
AGENT_FAILED_BUILDS_THR = 3 # agents declaired unhealthy if they have failed jobs from >=3 unique builds
RECIPIENTS = ["hissu.hyvarinen@silo.ai"]#, "olga.miroshnichenko@silo.ai"]

params = {
    'created_from': TODAY,
}

def fetch_data(params, token=BUILDKITE_API_TOKEN, org_slug=ORGANIZATION_SLUG, pipe_slug=PIPELINE_SLUG):
    # Set the URL
    url = f"https://api.buildkite.com/v2/organizations/{org_slug}/pipelines/{pipe_slug}/builds"


    # Set the headers
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    # Make the GET request
    response = requests.get(url, headers=headers, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        return pd.json_normalize(data)
    else:
        print(f"Request failed with status code {response.status_code}")
        return pd.DataFrame()

df = fetch_data(params)

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

def alert(df, wait_time_thr=WAITING_TIME_ALERT_THR, agent_failed_builds_thr=AGENT_FAILED_BUILDS_THR):
    alerts = []
    for index, row in df.iterrows():
        if row['waited_seconds'] > wait_time_thr:
            alert_message = f"Job {row['name']} from build number {row['number']} waited for {row['waited_seconds']} seconds (more than {wait_time_thr} or {wait_time_thr/3600} hours). More info at {row['web_url_job']}"
            alerts.append(alert_message)
        if (row['state_job'] == 'failed') & (row['soft_failed'] == False): 
            alert_message = f"Job {row['name']} from build number {row['number']} failed. More info at {row['web_url_job']}"
            alerts.append(alert_message)

    # alert for agent health:
    failed_jobs_from_diff_builds = df[(df.state_job=='failed') & (df.soft_failed==False)].groupby(['agent.id', 'agent.name', 'agent.web_url'], as_index=False).agg(unique_builds=('id_build', 'nunique'))
    unhealthy_agents = failed_jobs_from_diff_builds[failed_jobs_from_diff_builds.unique_builds>=agent_failed_builds_thr]
    for index, row in unhealthy_agents.iterrows():
        alert_message = f"Agent {row['agent.name']} has failed jobs from {row['unique_builds']} unique builds. More info at {row['agent.web_url']}"
        alerts.append(alert_message)


    return alerts

alerts = alert(result_df_amd)


def send_email(alerts, recipients=RECIPIENTS):
    # Sends email using gmail's username and app password that are in credentials.txt
    # in the format username:password

    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    try:
        s.login(GMAIL_USERNAME, GMAIL_PASSWORD)
    except smtplib.SMTPAuthenticationError:
        print("SMTP authentication failed")
        # What do we want to do with the alert data if this happens?
        raise SystemExit


    msg = MIMEMultipart()
    msg['Subject'] = "Test email"
    msg['From'] = GMAIL_USERNAME
    msg['To'] = ", ".join(recipients)

    alerts_str = '\n'.join(alerts)
    msg.attach(MIMEText(alerts_str, 'plain'))

    s.send_message(msg)
    s.quit()



if alerts:
    print('sending email')
    send_email(alerts)  
else:
    print('No alerts this time')      