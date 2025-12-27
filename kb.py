import requests
import json
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ServiceNow credentials
SERVICENOW_INSTANCE = "https://dev319024.service-now.com"
USERNAME = "admin"
PASSWORD = "/r+j7u8eVVUQ"     # <-- replace with your actual pwd

# API endpoint for KB articles
url = f"{SERVICENOW_INSTANCE}/api/now/table/kb_knowledge"

# Query parameters (optional)
params = {
    "sysparm_limit": "2000",              # number of articles to fetch
    "sysparm_fields": "number,short_description",  
    "sysparm_query": "active=true"      # filter only active articles
}

# API call
response = requests.get(
    url,
    auth=(USERNAME, PASSWORD),
    headers={"Accept": "application/json"},
    params=params,
    verify=False
)

# Check response
if response.status_code == 200:
    data = response.json()
    print("Fetched KB Articles:")
    print(json.dumps(data, indent=4))
else:
    print("Error:", response.status_code)
    print(response.text)
