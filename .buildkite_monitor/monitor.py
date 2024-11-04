import pandas as pd
from dotenv import load_dotenv

import os
import json
import requests
from datetime import datetime

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# Load environment variables from .env file
load_dotenv()

BUILDKITE_API_TOKEN = os.getenv('BUILDKITE_API_TOKEN')
ORGANIZATION_SLUG='vllm'
PIPELINE_SLUG = 'ci-aws'
TODAY = datetime.utcnow().strftime('%Y-%m-%dT00:00:00Z') # it is UTC, so -2 hours from Finnish local time

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
