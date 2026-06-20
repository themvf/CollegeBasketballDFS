"""
Apply lifecycle rules to the GCS bucket to move old objects to cheaper storage classes.

Storage class pricing (per GB/month):
  Standard:  $0.020  (current)
  Nearline:  $0.010  (accessed < 1×/month, min 30-day storage)
  Coldline:  $0.004  (accessed < 1×/quarter, min 90-day storage)
  Archive:   $0.0012 (accessed < 1×/year, min 365-day storage)

Run once:
  python scripts/set_gcs_lifecycle.py

Env vars used (same as rest of app):
  GCP_SERVICE_ACCOUNT_JSON or GCP_SERVICE_ACCOUNT_JSON_B64
  CBB_GCS_BUCKET  (default: collegebasketballdfs)
  GCP_PROJECT
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from college_basketball_dfs.cbb_gcs import build_storage_client


def main() -> None:
    bucket_name = os.environ.get("CBB_GCS_BUCKET", "collegebasketballdfs")
    project = os.environ.get("GCP_PROJECT")

    client = build_storage_client(project=project)
    bucket = client.bucket(bucket_name)
    bucket.reload()

    rules = [
        # Transition to Nearline after 30 days (50% cheaper, read latency identical)
        {
            "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
            "condition": {"age": 30},
        },
        # Transition to Coldline after 90 days (80% cheaper)
        {
            "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
            "condition": {"age": 90},
        },
        # Transition to Archive after 365 days (94% cheaper, 12-hour retrieval SLA)
        {
            "action": {"type": "SetStorageClass", "storageClass": "ARCHIVE"},
            "condition": {"age": 365},
        },
        # Delete non-current object versions after 30 days (cleans up overwrites)
        {
            "action": {"type": "Delete"},
            "condition": {"age": 30, "isLive": False},
        },
    ]

    bucket.lifecycle_rules = rules
    bucket.patch()
    print(f"Lifecycle rules applied to gs://{bucket_name}")
    for rule in rules:
        cond = rule["condition"]
        act = rule["action"]
        print(f"  age>{cond.get('age')}d → {act['type']} {act.get('storageClass', '')}")


if __name__ == "__main__":
    main()
