import pandas as pd
import numpy as np
from dotenv import load_dotenv

import os
import json
import requests
from datetime import datetime
import zoneinfo
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
WAITING_TIME_ALERT_THR = 1#14400 # 4 hours
AGENT_FAILED_BUILDS_THR = 1#3 # agents declaired unhealthy if they have failed jobs from >=3 unique builds
RECIPIENTS = ['olga.miroshnichenko@silo.ai']#["hissu.hyvarinen@silo.ai", "olga.miroshnichenko@silo.ai", 'alexei_v_ivanov@ieee.org']
PATH_TO_LOGS = '/mnt/home/buildkite_logs/'

params = {
    'created_from': TODAY,
    'per_page': 100,
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

def write_fetch_log(df, path=PATH_TO_LOGS):
    '''Writing logs for testing purposes'''
    df.to_csv(path + 'fetch_' + datetime.now(zoneinfo.ZoneInfo('Europe/Helsinki')).isoformat() + '.csv')
    return None

write_fetch_log(df)

def types_fix(df, jobs=False):
    df['number'] = df['number'].astype('int')
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['scheduled_at'] = pd.to_datetime(df['scheduled_at'], errors='coerce')
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')   
    df['finished_at'] = pd.to_datetime(df['finished_at'], errors='coerce')
    if jobs:
        df['soft_failed'] = df['soft_failed'].astype('bool')
        df['runnable_at'] = pd.to_datetime(df['runnable_at'], errors='coerce')
        df['expired_at'] = pd.to_datetime(df['expired_at'], errors='coerce')
        df.columns = df.columns.str.replace('.', '_')


    return df

df = types_fix(df, jobs=False)



useful_columns = ['id', 'web_url', 'url', 'number', 'state', 'cancel_reason', 'blocked', 'blocked_state', 'jobs']
d = df[useful_columns]

jobs_df = pd.json_normalize(df['jobs'].explode())
jobs_useful_columns = ['id', 'name', 'state', 'build_url', 'web_url', 'soft_failed', 'created_at', 'scheduled_at', 'runnable_at', 'started_at',	'finished_at', 'expired_at', 'retried', 'agent.id', 'agent.name', 'agent.web_url', 'agent.connection_state', 'agent.meta_data']
j = jobs_df[jobs_useful_columns]

result_df = d.drop(columns=['jobs']).merge(j.reset_index(drop=True), left_on='url', right_on='build_url', suffixes=['_build', '_job'], how='outer')

# filter to AMD tests:
result_df_amd = result_df[(result_df.name.notna()) & (result_df.name.str.contains('AMD'))]

result_df_amd = types_fix(result_df_amd, jobs=True)

def calculate_wait_time(df):
    now_utc = pd.Timestamp.now(tz='UTC')
    
    # Calculate the difference in seconds
    df['waited_seconds'] = df.apply(
        lambda row: (row['started_at'] - row['runnable_at']).total_seconds() if pd.notna(row['started_at']) and pd.notna(row['runnable_at']) \
              else (now_utc - row['runnable_at']).total_seconds() if pd.isna(row['started_at']) and pd.notna(row['runnable_at']) \
                else None,
        axis=1
    )
    
    return df


result_df_amd = calculate_wait_time(result_df_amd)

try:
    alerts_sent = pd.read_csv(PATH_TO_LOGS + 'alerts_sent.csv')
except:
    alerts_sent = pd.DataFrame()

def alert(df, alerts_sent=alerts_sent, wait_time_thr=WAITING_TIME_ALERT_THR, agent_failed_builds_thr=AGENT_FAILED_BUILDS_THR):
    alerts = []
    now = datetime.now().isoformat()
    alerts_df = pd.DataFrame(columns=['time_utc','alert_type', 'id_job', 'state_job', 'name', 'number', 'id_build', 'waited_seconds', 'web_url_job', 'agent_id', 'agent_name', 'agent_web_url', 'nunique_failed_builds', 'failed_builds'] )
    
    # job waiting time alert:
    for _, row in df.iterrows():
        if row['waited_seconds'] > wait_time_thr and row['state_job']!='canceled' and pd.isna(row['started_at']): 
            if not alerts_sent.empty:
                value_exists_in_column = alerts_sent['id_job'].isin([row['id_job']]).any()
                if value_exists_in_column:
                    continue
            new_row = pd.DataFrame.from_records({'time_utc': now, 'alert_type': 'job', 'id_job': row['id_job'], 'state_job': row['state_job'], 'name': row['name'], 'number': row['number'], 'id_build': row['id_build'], 'waited_seconds': row['waited_seconds'],  
                       'web_url_job': row['web_url_job'], 'agent_id': row['agent_id'], 'agent_name': row['agent_name'], 'agent_web_url': row['agent_web_url'], 'nunique_failed_builds': np.nan, 'failed_builds': [[]]}) 
            alert_message = f"Job {row['name']} from build number {row['number']} has been waiting for {row['waited_seconds']} seconds (more than {wait_time_thr} seconds or {wait_time_thr/3600} hours). More info at {row['web_url_job']}"
            alerts.append(alert_message)
            alerts_df = pd.concat([alerts_df, new_row], ignore_index=True)
    
    
    # agent health alert:
    failed_jobs_from_diff_builds = df[(df.state_job=='failed') & (df.soft_failed==False)].groupby(['agent_id', 'agent_name', 'agent_web_url'], as_index=False).agg(nunique_failed_builds=('id_build', 'nunique'), failed_builds=('id_build', 'unique'))
    
    unhealthy_agents = failed_jobs_from_diff_builds[failed_jobs_from_diff_builds.nunique_failed_builds>=agent_failed_builds_thr]
    for _, row in unhealthy_agents.iterrows():
        if not alerts_sent.empty:
                value_exists_in_column = alerts_sent['agent_id'].isin([row['agent_id']]).any()                
                if value_exists_in_column:
                    sent_failed_builds = alerts_sent.loc[alerts_sent['agent_id'] == row['agent_id'], 'failed_builds'].str.findall(r'([a-f0-9\-]{36})').values[0]
        
                    # Check if there is any intersection between row['failed_builds'] and sent_failed_builds
                    if any(failed_build in sent_failed_builds for failed_build in row['failed_builds']):
                        continue
        new_row = pd.DataFrame({'time_utc': now, 'alert_type': 'agent', 'id_job': np.nan, 'state_job': np.nan, 'name': np.nan, 'number': np.nan, 'id_build': np.nan, 'waited_seconds': np.nan,  
                        'web_url_job': np.nan, 'agent_id': row['agent_id'], 'agent_name': row['agent_name'], 'agent_web_url': row['agent_web_url'], 'nunique_failed_builds': row['nunique_failed_builds'], 'failed_builds': [row['failed_builds'].tolist()]}, index=[0])
        
        alert_message = f"Agent {row['agent_name']} has failed jobs from {row['nunique_failed_builds']} unique builds. More info at {row['agent_web_url']}"
        alerts.append(alert_message)
        alerts_df = pd.concat([alerts_df, new_row], ignore_index=True)

    return alerts, alerts_df

alerts, alerts_df = alert(result_df_amd)


def send_email(alerts, alerts_df, recipients=RECIPIENTS):
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
    alerts_df.to_csv(PATH_TO_LOGS + 'alerts_sent.csv')



if alerts:
    print('sending email')
    send_email(alerts, alerts_df)  
else:
    print('No alerts this time')      