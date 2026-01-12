import requests
import json
import time

# Define the API endpoint URL
url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

# Define the query parameters
query_params = {
    "query": '"phlogiston"',
    "fields": "title,url,publicationTypes,publicationDate,openAccessPdf",
    "year": "1900-1950"
}

# Directly define the API key (Reminder: Securely handle API keys in production environments)
api_key = ""  # TODO: change for ENV variable

# Define headers with API key
headers = {"x-api-key": api_key}

# Send the API request

import time

def send_request_with_retries(url, params, headers, retries=3):
    for attempt in range(retries):
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:  # Too Many Requests
            retry_after = int(response.headers.get("Retry-After", 1))
            time.sleep(retry_after)
        else:
            time.sleep(2 ** attempt)  # Exponential backoff
    response.raise_for_status()

response = send_request_with_retries(url, params=query_params, headers=headers)

print(f"Will retrieve an estimated {response['total']} documents")
retrieved = 0

# Write results to json file and get next batch of results
with open(f"papers.json", "a") as file:
    while True:
        if "data" in response:
            retrieved += len(response["data"])
            print(f"Retrieved {retrieved} papers...")
            for paper in response["data"]:
                print(json.dumps(paper), file=file)
        # checks for continuation token to get next batch of results
        if "token" not in response:
            break
        response = requests.get(f"{url}&token={response['token']}").json()

print(f"Done! Retrieved {retrieved} papers total")