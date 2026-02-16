from __future__ import annotations

import base64
import json
import os
import re
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
    dk_slates_prefix: str = "cbb/dk_slates"
    injuries_prefix: str = "cbb/injuries"
    projections_prefix: str = "cbb/projections"
    ownership_prefix: str = "cbb/ownership"
    contest_standings_prefix: str = "cbb/contest_standings"
    lineup_runs_prefix: str = "cbb/lineup_runs"
    phantom_reviews_prefix: str = "cbb/phantom_reviews"

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

    def dk_slate_blob_name(self, game_date: date) -> str:
        return f"{self.dk_slates_prefix}/{game_date.isoformat()}_dk_slate.csv"

    def injuries_blob_name(self) -> str:
        return f"{self.injuries_prefix}/injuries_master.csv"

    def injuries_feed_blob_name(self) -> str:
        return f"{self.injuries_prefix}/injuries_feed.csv"

    def injuries_manual_blob_name(self) -> str:
        return f"{self.injuries_prefix}/injuries_manual.csv"

    def projections_blob_name(self, game_date: date) -> str:
        return f"{self.projections_prefix}/{game_date.isoformat()}_projections.csv"

    def ownership_blob_name(self, game_date: date) -> str:
        return f"{self.ownership_prefix}/{game_date.isoformat()}_ownership.csv"

    def contest_standings_blob_name(self, game_date: date, contest_id: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(contest_id or "").strip())
        safe = safe or "contest"
        return f"{self.contest_standings_prefix}/{game_date.isoformat()}_{safe}.csv"

    def _safe_key(self, value: str, default: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(value or "").strip())
        return safe or default

    def lineup_run_prefix(self, game_date: date, run_id: str) -> str:
        run_key = self._safe_key(run_id, "run")
        return f"{self.lineup_runs_prefix}/{game_date.isoformat()}/{run_key}"

    def lineup_run_manifest_blob_name(self, game_date: date, run_id: str) -> str:
        return f"{self.lineup_run_prefix(game_date, run_id)}/manifest.json"

    def lineup_version_json_blob_name(self, game_date: date, run_id: str, version_key: str) -> str:
        safe_version = self._safe_key(version_key, "version")
        return f"{self.lineup_run_prefix(game_date, run_id)}/{safe_version}/lineups.json"

    def lineup_version_csv_blob_name(self, game_date: date, run_id: str, version_key: str) -> str:
        safe_version = self._safe_key(version_key, "version")
        return f"{self.lineup_run_prefix(game_date, run_id)}/{safe_version}/lineups.csv"

    def lineup_version_upload_blob_name(self, game_date: date, run_id: str, version_key: str) -> str:
        safe_version = self._safe_key(version_key, "version")
        return f"{self.lineup_run_prefix(game_date, run_id)}/{safe_version}/dk_upload.csv"

    def phantom_review_csv_blob_name(self, game_date: date, run_id: str, version_key: str) -> str:
        run_key = self._safe_key(run_id, "run")
        version = self._safe_key(version_key, "version")
        return f"{self.phantom_reviews_prefix}/{game_date.isoformat()}/{run_key}/{version}.csv"

    def phantom_review_summary_blob_name(self, game_date: date, run_id: str) -> str:
        run_key = self._safe_key(run_id, "run")
        return f"{self.phantom_reviews_prefix}/{game_date.isoformat()}/{run_key}/summary.json"

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

    def list_players_blob_names(self) -> list[str]:
        blobs = self.bucket.list_blobs(prefix=f"{self.players_prefix}/")
        names = [blob.name for blob in blobs if blob.name.endswith("_players.csv")]
        names.sort()
        return names

    def read_raw_json_blob(self, blob_name: str) -> dict[str, Any]:
        blob = self.bucket.blob(blob_name)
        text = blob.download_as_text(encoding="utf-8")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected cached payload type in {blob_name}: {type(payload).__name__}")
        return payload

    def read_players_csv_blob(self, blob_name: str) -> str:
        blob = self.bucket.blob(blob_name)
        return blob.download_as_text(encoding="utf-8")

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

    def read_dk_slate_csv(self, game_date: date) -> str | None:
        blob = self.bucket.blob(self.dk_slate_blob_name(game_date))
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")

    def write_dk_slate_csv(self, game_date: date, csv_text: str) -> str:
        blob_name = self.dk_slate_blob_name(game_date)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def delete_dk_slate_csv(self, game_date: date) -> bool:
        blob = self.bucket.blob(self.dk_slate_blob_name(game_date))
        if not blob.exists():
            return False
        blob.delete()
        return True

    def read_injuries_csv(self) -> str | None:
        blob = self.bucket.blob(self.injuries_blob_name())
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")

    def write_injuries_csv(self, csv_text: str) -> str:
        blob_name = self.injuries_blob_name()
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def read_injuries_feed_csv(self) -> str | None:
        blob = self.bucket.blob(self.injuries_feed_blob_name())
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")

    def write_injuries_feed_csv(self, csv_text: str) -> str:
        blob_name = self.injuries_feed_blob_name()
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def read_injuries_manual_csv(self) -> str | None:
        blob = self.bucket.blob(self.injuries_manual_blob_name())
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")

    def write_injuries_manual_csv(self, csv_text: str) -> str:
        blob_name = self.injuries_manual_blob_name()
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def read_projections_csv(self, game_date: date) -> str | None:
        blob = self.bucket.blob(self.projections_blob_name(game_date))
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")

    def write_projections_csv(self, game_date: date, csv_text: str) -> str:
        blob_name = self.projections_blob_name(game_date)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def read_ownership_csv(self, game_date: date) -> str | None:
        blob = self.bucket.blob(self.ownership_blob_name(game_date))
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")

    def write_ownership_csv(self, game_date: date, csv_text: str) -> str:
        blob_name = self.ownership_blob_name(game_date)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def read_contest_standings_csv(self, game_date: date, contest_id: str) -> str | None:
        blob = self.bucket.blob(self.contest_standings_blob_name(game_date, contest_id))
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")

    def write_contest_standings_csv(self, game_date: date, contest_id: str, csv_text: str) -> str:
        blob_name = self.contest_standings_blob_name(game_date, contest_id)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def read_lineup_run_manifest_json(self, game_date: date, run_id: str) -> dict[str, Any] | None:
        blob = self.bucket.blob(self.lineup_run_manifest_blob_name(game_date, run_id))
        if not blob.exists():
            return None
        payload = json.loads(blob.download_as_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected lineup-run manifest payload type: {type(payload).__name__}")
        return payload

    def write_lineup_run_manifest_json(self, game_date: date, run_id: str, payload: dict[str, Any]) -> str:
        blob_name = self.lineup_run_manifest_blob_name(game_date, run_id)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(payload, indent=2), content_type="application/json")
        return blob_name

    def read_lineup_version_json(self, game_date: date, run_id: str, version_key: str) -> dict[str, Any] | None:
        blob = self.bucket.blob(self.lineup_version_json_blob_name(game_date, run_id, version_key))
        if not blob.exists():
            return None
        payload = json.loads(blob.download_as_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected lineup version payload type: {type(payload).__name__}")
        return payload

    def write_lineup_version_json(
        self,
        game_date: date,
        run_id: str,
        version_key: str,
        payload: dict[str, Any],
    ) -> str:
        blob_name = self.lineup_version_json_blob_name(game_date, run_id, version_key)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(payload, indent=2), content_type="application/json")
        return blob_name

    def read_lineup_version_csv(self, game_date: date, run_id: str, version_key: str) -> str | None:
        blob = self.bucket.blob(self.lineup_version_csv_blob_name(game_date, run_id, version_key))
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")

    def write_lineup_version_csv(self, game_date: date, run_id: str, version_key: str, csv_text: str) -> str:
        blob_name = self.lineup_version_csv_blob_name(game_date, run_id, version_key)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def read_lineup_version_upload_csv(self, game_date: date, run_id: str, version_key: str) -> str | None:
        blob = self.bucket.blob(self.lineup_version_upload_blob_name(game_date, run_id, version_key))
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")

    def write_lineup_version_upload_csv(self, game_date: date, run_id: str, version_key: str, csv_text: str) -> str:
        blob_name = self.lineup_version_upload_blob_name(game_date, run_id, version_key)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def list_lineup_run_ids(self, game_date: date) -> list[str]:
        prefix = f"{self.lineup_runs_prefix}/{game_date.isoformat()}/"
        blobs = self.bucket.list_blobs(prefix=prefix)
        run_ids: set[str] = set()
        for blob in blobs:
            suffix = str(blob.name or "")[len(prefix) :]
            if not suffix:
                continue
            run_id = suffix.split("/", 1)[0].strip()
            if run_id:
                run_ids.add(run_id)
        out = sorted(run_ids, reverse=True)
        return out

    def list_lineup_run_dates(self) -> list[date]:
        prefix = f"{self.lineup_runs_prefix}/"
        blobs = self.bucket.list_blobs(prefix=prefix)
        dates_found: set[date] = set()
        for blob in blobs:
            suffix = str(blob.name or "")[len(prefix) :]
            if not suffix:
                continue
            date_part = suffix.split("/", 1)[0].strip()
            if not date_part:
                continue
            try:
                parsed = date.fromisoformat(date_part)
            except ValueError:
                continue
            dates_found.add(parsed)
        return sorted(dates_found, reverse=True)

    def read_phantom_review_csv(self, game_date: date, run_id: str, version_key: str) -> str | None:
        blob = self.bucket.blob(self.phantom_review_csv_blob_name(game_date, run_id, version_key))
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")

    def write_phantom_review_csv(self, game_date: date, run_id: str, version_key: str, csv_text: str) -> str:
        blob_name = self.phantom_review_csv_blob_name(game_date, run_id, version_key)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def read_phantom_review_summary_json(self, game_date: date, run_id: str) -> dict[str, Any] | None:
        blob = self.bucket.blob(self.phantom_review_summary_blob_name(game_date, run_id))
        if not blob.exists():
            return None
        payload = json.loads(blob.download_as_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected phantom-review summary payload type: {type(payload).__name__}")
        return payload

    def write_phantom_review_summary_json(self, game_date: date, run_id: str, payload: dict[str, Any]) -> str:
        blob_name = self.phantom_review_summary_blob_name(game_date, run_id)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(payload, indent=2), content_type="application/json")
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
