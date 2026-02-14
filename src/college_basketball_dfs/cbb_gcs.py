from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Any

try:
    from google.cloud import storage  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - handled at runtime in build_storage_client
    storage = None  # type: ignore[assignment]


def build_storage_client(
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
    project: str | None = None,
) -> Any:
    """Build a Google Cloud Storage client from JSON creds or ADC."""
    if storage is None:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "google-cloud-storage is not installed. Install dependency or use requirements.txt setup."
        )

    raw_json = (
        service_account_json
        or os.getenv("GCP_SERVICE_ACCOUNT_JSON")
        or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    )
    raw_b64 = service_account_json_b64 or os.getenv("GCP_SERVICE_ACCOUNT_JSON_B64")

    if raw_json:
        info = json.loads(raw_json)
        return storage.Client.from_service_account_info(info, project=project)

    if raw_b64:
        decoded = base64.b64decode(raw_b64).decode("utf-8")
        info = json.loads(decoded)
        return storage.Client.from_service_account_info(info, project=project)

    return storage.Client(project=project)


@dataclass(frozen=True)
class CbbGcsStore:
    bucket_name: str
    client: Any
    raw_prefix: str = "cbb/raw"
    players_prefix: str = "cbb/players"
    odds_prefix: str = "cbb/odds"
    odds_games_prefix: str = "cbb/odds_games"
    props_prefix: str = "cbb/props"
    props_lines_prefix: str = "cbb/props_lines"

    def __post_init__(self) -> None:
        if not self.bucket_name:
            raise ValueError("bucket_name is required for CbbGcsStore.")
        if self.client is None:
            raise ValueError("client is required for CbbGcsStore.")

    @property
    def bucket(self) -> Any:
        return self.client.bucket(self.bucket_name)

    def raw_blob_name(self, game_date: date) -> str:
        return f"{self.raw_prefix}/{game_date.isoformat()}.json"

    def players_blob_name(self, game_date: date) -> str:
        return f"{self.players_prefix}/{game_date.isoformat()}_players.csv"

    def odds_blob_name(self, game_date: date) -> str:
        return f"{self.odds_prefix}/{game_date.isoformat()}.json"

    def odds_games_blob_name(self, game_date: date) -> str:
        return f"{self.odds_games_prefix}/{game_date.isoformat()}_odds.csv"

    def props_blob_name(self, game_date: date) -> str:
        return f"{self.props_prefix}/{game_date.isoformat()}.json"

    def props_lines_blob_name(self, game_date: date) -> str:
        return f"{self.props_lines_prefix}/{game_date.isoformat()}_props.csv"

    def read_raw_json(self, game_date: date) -> dict[str, Any] | None:
        blob = self.bucket.blob(self.raw_blob_name(game_date))
        if not blob.exists():
            return None
        text = blob.download_as_text(encoding="utf-8")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected cached payload type: {type(payload).__name__}")
        return payload

    def list_raw_blob_names(self) -> list[str]:
        blobs = self.bucket.list_blobs(prefix=f"{self.raw_prefix}/")
        names = [blob.name for blob in blobs if blob.name.endswith(".json")]
        names.sort()
        return names

    def read_raw_json_blob(self, blob_name: str) -> dict[str, Any]:
        blob = self.bucket.blob(blob_name)
        text = blob.download_as_text(encoding="utf-8")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected cached payload type in {blob_name}: {type(payload).__name__}")
        return payload

    def read_odds_json(self, game_date: date) -> dict[str, Any] | None:
        blob = self.bucket.blob(self.odds_blob_name(game_date))
        if not blob.exists():
            return None
        text = blob.download_as_text(encoding="utf-8")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected odds payload type: {type(payload).__name__}")
        return payload

    def write_odds_json(self, game_date: date, payload: dict[str, Any]) -> str:
        blob_name = self.odds_blob_name(game_date)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(payload, indent=2),
            content_type="application/json",
        )
        return blob_name

    def write_odds_games_csv(self, game_date: date, csv_text: str) -> str:
        blob_name = self.odds_games_blob_name(game_date)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def read_props_json(self, game_date: date) -> dict[str, Any] | None:
        blob = self.bucket.blob(self.props_blob_name(game_date))
        if not blob.exists():
            return None
        text = blob.download_as_text(encoding="utf-8")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected props payload type: {type(payload).__name__}")
        return payload

    def write_props_json(self, game_date: date, payload: dict[str, Any]) -> str:
        blob_name = self.props_blob_name(game_date)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(payload, indent=2),
            content_type="application/json",
        )
        return blob_name

    def write_props_lines_csv(self, game_date: date, csv_text: str) -> str:
        blob_name = self.props_lines_blob_name(game_date)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def write_raw_json(self, game_date: date, payload: dict[str, Any]) -> str:
        blob_name = self.raw_blob_name(game_date)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(payload, indent=2),
            content_type="application/json",
        )
        return blob_name

    def write_players_csv(self, game_date: date, csv_text: str) -> str:
        blob_name = self.players_blob_name(game_date)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name
