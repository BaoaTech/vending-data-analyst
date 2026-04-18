import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

QB_REALM = os.environ["QB_REALM"] + ".quickbase.com"
QB_TOKEN = os.environ["QB_USER_TOKEN"]


def api_load_data(base_body, batch_size=50000, sleep=0.2):
    headers = {
        'Content-Type': 'application/json',
        'QB-Realm-Hostname': QB_REALM,
        'User-Agent': 'QB Master',
        'Authorization': f'QB-USER-TOKEN {QB_TOKEN}'
    }

    all_records = []
    skip = 0

    while True:
        body = base_body.copy()
        body["options"] = {"skip": skip, "top": batch_size}

        r = requests.post('https://api.quickbase.com/v1/records/query', headers=headers, json=body)
        r.raise_for_status()
        records = r.json().get("data", [])

        if not records:
            break

        all_records.extend(records)
        skip += batch_size
        print(f"📥 Descargados: {len(all_records)} registros")
        time.sleep(sleep)

    return all_records


def get_fields(table_id):
    headers = {
        'QB-Realm-Hostname': QB_REALM,
        'Authorization': f'QB-USER-TOKEN {QB_TOKEN}'
    }
    r = requests.get(
        'https://api.quickbase.com/v1/fields',
        headers=headers,
        params={'tableId': table_id}
    )
    r.raise_for_status()
    return {str(f['id']): f['label'] for f in r.json()}

