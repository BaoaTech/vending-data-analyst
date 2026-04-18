import os
from datetime import datetime
import pandas as pd


def save_snapshot(df, name, base_dir=None):
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "snapshots")
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"{name}_{timestamp}.csv")
    df.to_csv(path, index=False)
    print(f"💾 Snapshot guardado: {path}")
    return path


def parser_create_df(records):
    parsed_rows = []

    for record in records:
        row = {}
        for field_id, field_data in record.items():
            row[field_id] = field_data.get("value")
        parsed_rows.append(row)

    return pd.DataFrame(parsed_rows)
