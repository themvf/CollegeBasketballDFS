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

    def dk_slate_blob_name(self, game_date: date, slate_key: str | None = None) -> str:
        if slate_key is None:
            # Legacy single-file-per-date path.
            return f"{self.dk_slates_prefix}/{game_date.isoformat()}_dk_slate.csv"
        safe_slate = self._safe_slate_key(slate_key, default="main")
        return f"{self.dk_slates_prefix}/{game_date.isoformat()}/{safe_slate}_dk_slate.csv"

    def injuries_blob_name(self) -> str:
        return f"{self.injuries_prefix}/injuries_master.csv"

    def injuries_feed_blob_name(self, game_date: date | None = None) -> str:
        if game_date is None:
            return f"{self.injuries_prefix}/injuries_feed.csv"
        return f"{self.injuries_prefix}/feed/{game_date.isoformat()}_injuries_feed.csv"

    def injuries_manual_blob_name(self) -> str:
        return f"{self.injuries_prefix}/injuries_manual.csv"

    def projections_blob_name(self, game_date: date, slate_key: str | None = None) -> str:
        if slate_key is None:
            # Legacy date-only path.
            return f"{self.projections_prefix}/{game_date.isoformat()}_projections.csv"
        safe_slate = self._safe_slate_key(slate_key, default="main")
        return f"{self.projections_prefix}/{game_date.isoformat()}/{safe_slate}_projections.csv"

    def ownership_blob_name(self, game_date: date, slate_key: str | None = None) -> str:
        if slate_key is None:
            # Legacy date-only path.
            return f"{self.ownership_prefix}/{game_date.isoformat()}_ownership.csv"
        safe_slate = self._safe_slate_key(slate_key, default="main")
        return f"{self.ownership_prefix}/{game_date.isoformat()}/{safe_slate}_ownership.csv"

    def contest_standings_blob_name(
        self,
        game_date: date,
        contest_id: str,
        slate_key: str | None = None,
    ) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(contest_id or "").strip())
        safe = safe or "contest"
        if slate_key is None:
            # Legacy date-only path.
            return f"{self.contest_standings_prefix}/{game_date.isoformat()}_{safe}.csv"
        safe_slate = self._safe_slate_key(slate_key, default="main")
        return f"{self.contest_standings_prefix}/{game_date.isoformat()}/{safe_slate}_{safe}.csv"

    def _safe_key(self, value: str, default: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(value or "").strip())
        return safe or default

    def _safe_slate_key(self, value: str | None, default: str = "main") -> str:
        safe = self._safe_key(str(value or ""), default=default)
        return safe.lower()

    def lineup_run_prefix(self, game_date: date, run_id: str, slate_key: str | None = None) -> str:
        run_key = self._safe_key(run_id, "run")
        if slate_key is None:
            # Legacy date/run path.
            return f"{self.lineup_runs_prefix}/{game_date.isoformat()}/{run_key}"
        safe_slate = self._safe_slate_key(slate_key, default="main")
        return f"{self.lineup_runs_prefix}/{game_date.isoformat()}/{safe_slate}/{run_key}"

    def lineup_run_manifest_blob_name(self, game_date: date, run_id: str, slate_key: str | None = None) -> str:
        return f"{self.lineup_run_prefix(game_date, run_id, slate_key=slate_key)}/manifest.json"

    def lineup_version_json_blob_name(
        self,
        game_date: date,
        run_id: str,
        version_key: str,
        slate_key: str | None = None,
    ) -> str:
        safe_version = self._safe_key(version_key, "version")
        return f"{self.lineup_run_prefix(game_date, run_id, slate_key=slate_key)}/{safe_version}/lineups.json"

    def lineup_version_csv_blob_name(
        self,
        game_date: date,
        run_id: str,
        version_key: str,
        slate_key: str | None = None,
    ) -> str:
        safe_version = self._safe_key(version_key, "version")
        return f"{self.lineup_run_prefix(game_date, run_id, slate_key=slate_key)}/{safe_version}/lineups.csv"

    def lineup_version_upload_blob_name(
        self,
        game_date: date,
        run_id: str,
        version_key: str,
        slate_key: str | None = None,
    ) -> str:
        safe_version = self._safe_key(version_key, "version")
        return f"{self.lineup_run_prefix(game_date, run_id, slate_key=slate_key)}/{safe_version}/dk_upload.csv"

    def _find_lineup_blob_name(
        self,
        game_date: date,
        run_id: str,
        relative_suffix: str,
        slate_key: str | None = None,
    ) -> str | None:
        date_prefix = f"{self.lineup_runs_prefix}/{game_date.isoformat()}/"
        run_key = self._safe_key(run_id, "run")
        legacy_blob_name = f"{date_prefix}{run_key}/{relative_suffix}"

        if slate_key is not None:
            safe_slate = self._safe_slate_key(slate_key, default="main")
            preferred = f"{date_prefix}{safe_slate}/{run_key}/{relative_suffix}"
            preferred_blob = self.bucket.blob(preferred)
            if preferred_blob.exists():
                return preferred
            if safe_slate == "main":
                legacy_blob = self.bucket.blob(legacy_blob_name)
                if legacy_blob.exists():
                    return legacy_blob_name
            return None

        legacy_blob = self.bucket.blob(legacy_blob_name)
        if legacy_blob.exists():
            return legacy_blob_name

        search_suffix = f"/{run_key}/{relative_suffix}"
        matches = sorted(
            {
                str(blob.name or "")
                for blob in self.bucket.list_blobs(prefix=date_prefix)
                if str(blob.name or "").endswith(search_suffix)
            }
        )
        if not matches:
            return None
        return matches[0]

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

    def read_dk_slate_csv(self, game_date: date, slate_key: str | None = None) -> str | None:
        if slate_key is None:
            candidate_names = [
                self.dk_slate_blob_name(game_date, slate_key="main"),
                self.dk_slate_blob_name(game_date),
            ]
        else:
            candidate_names = [self.dk_slate_blob_name(game_date, slate_key=slate_key)]
            safe_slate = self._safe_slate_key(slate_key, default="main")
            # Backward compatibility: allow reads of legacy date-only blob for main slate.
            if safe_slate == "main":
                candidate_names.append(self.dk_slate_blob_name(game_date))
        for blob_name in candidate_names:
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                return blob.download_as_text(encoding="utf-8")
        return None

    def write_dk_slate_csv(self, game_date: date, csv_text: str, slate_key: str | None = None) -> str:
        blob_name = self.dk_slate_blob_name(game_date, slate_key=slate_key)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def delete_dk_slate_csv(self, game_date: date, slate_key: str | None = None) -> bool:
        if slate_key is None:
            candidate_names = [
                self.dk_slate_blob_name(game_date, slate_key="main"),
                self.dk_slate_blob_name(game_date),
            ]
        else:
            candidate_names = [self.dk_slate_blob_name(game_date, slate_key=slate_key)]
            safe_slate = self._safe_slate_key(slate_key, default="main")
            if safe_slate == "main":
                candidate_names.append(self.dk_slate_blob_name(game_date))
        for blob_name in candidate_names:
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                blob.delete()
                return True
        return False

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

    def read_injuries_feed_csv(self, game_date: date | None = None) -> str | None:
        blob = self.bucket.blob(self.injuries_feed_blob_name(game_date))
        if not blob.exists():
            if game_date is not None:
                # Backward compatibility for older single-file feed storage.
                legacy_blob = self.bucket.blob(self.injuries_feed_blob_name(None))
                if legacy_blob.exists():
                    return legacy_blob.download_as_text(encoding="utf-8")
            return None
        return blob.download_as_text(encoding="utf-8")

    def write_injuries_feed_csv(self, csv_text: str, game_date: date | None = None) -> str:
        blob_name = self.injuries_feed_blob_name(game_date)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def delete_injuries_feed_csv(self, game_date: date | None = None) -> bool:
        blob = self.bucket.blob(self.injuries_feed_blob_name(game_date))
        if not blob.exists():
            return False
        blob.delete()
        return True

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

    def read_projections_csv(self, game_date: date, slate_key: str | None = None) -> str | None:
        if slate_key is None:
            candidate_names = [
                self.projections_blob_name(game_date, slate_key="main"),
                self.projections_blob_name(game_date),
            ]
        else:
            candidate_names = [self.projections_blob_name(game_date, slate_key=slate_key)]
            safe_slate = self._safe_slate_key(slate_key, default="main")
            if safe_slate == "main":
                candidate_names.append(self.projections_blob_name(game_date))
        for blob_name in candidate_names:
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                return blob.download_as_text(encoding="utf-8")
        return None

    def write_projections_csv(self, game_date: date, csv_text: str, slate_key: str | None = None) -> str:
        blob_name = self.projections_blob_name(game_date, slate_key=slate_key)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def read_ownership_csv(self, game_date: date, slate_key: str | None = None) -> str | None:
        if slate_key is None:
            candidate_names = [
                self.ownership_blob_name(game_date, slate_key="main"),
                self.ownership_blob_name(game_date),
            ]
        else:
            candidate_names = [self.ownership_blob_name(game_date, slate_key=slate_key)]
            safe_slate = self._safe_slate_key(slate_key, default="main")
            if safe_slate == "main":
                candidate_names.append(self.ownership_blob_name(game_date))
        for blob_name in candidate_names:
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                return blob.download_as_text(encoding="utf-8")
        return None

    def write_ownership_csv(self, game_date: date, csv_text: str, slate_key: str | None = None) -> str:
        blob_name = self.ownership_blob_name(game_date, slate_key=slate_key)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def read_contest_standings_csv(
        self,
        game_date: date,
        contest_id: str,
        slate_key: str | None = None,
    ) -> str | None:
        if slate_key is None:
            candidate_names = [
                self.contest_standings_blob_name(game_date, contest_id, slate_key="main"),
                self.contest_standings_blob_name(game_date, contest_id),
            ]
        else:
            candidate_names = [self.contest_standings_blob_name(game_date, contest_id, slate_key=slate_key)]
            safe_slate = self._safe_slate_key(slate_key, default="main")
            if safe_slate == "main":
                candidate_names.append(self.contest_standings_blob_name(game_date, contest_id))
        for blob_name in candidate_names:
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                return blob.download_as_text(encoding="utf-8")
        return None

    def write_contest_standings_csv(
        self,
        game_date: date,
        contest_id: str,
        csv_text: str,
        slate_key: str | None = None,
    ) -> str:
        blob_name = self.contest_standings_blob_name(game_date, contest_id, slate_key=slate_key)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def read_lineup_run_manifest_json(
        self,
        game_date: date,
        run_id: str,
        slate_key: str | None = None,
    ) -> dict[str, Any] | None:
        blob_name = self._find_lineup_blob_name(
            game_date=game_date,
            run_id=run_id,
            relative_suffix="manifest.json",
            slate_key=slate_key,
        )
        if not blob_name:
            return None
        blob = self.bucket.blob(blob_name)
        payload = json.loads(blob.download_as_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected lineup-run manifest payload type: {type(payload).__name__}")
        return payload

    def write_lineup_run_manifest_json(
        self,
        game_date: date,
        run_id: str,
        payload: dict[str, Any],
        slate_key: str | None = None,
    ) -> str:
        blob_name = self.lineup_run_manifest_blob_name(game_date, run_id, slate_key=slate_key)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(payload, indent=2), content_type="application/json")
        return blob_name

    def read_lineup_version_json(
        self,
        game_date: date,
        run_id: str,
        version_key: str,
        slate_key: str | None = None,
    ) -> dict[str, Any] | None:
        safe_version = self._safe_key(version_key, "version")
        blob_name = self._find_lineup_blob_name(
            game_date=game_date,
            run_id=run_id,
            relative_suffix=f"{safe_version}/lineups.json",
            slate_key=slate_key,
        )
        if not blob_name:
            return None
        blob = self.bucket.blob(blob_name)
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
        slate_key: str | None = None,
    ) -> str:
        blob_name = self.lineup_version_json_blob_name(game_date, run_id, version_key, slate_key=slate_key)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(payload, indent=2), content_type="application/json")
        return blob_name

    def read_lineup_version_csv(
        self,
        game_date: date,
        run_id: str,
        version_key: str,
        slate_key: str | None = None,
    ) -> str | None:
        safe_version = self._safe_key(version_key, "version")
        blob_name = self._find_lineup_blob_name(
            game_date=game_date,
            run_id=run_id,
            relative_suffix=f"{safe_version}/lineups.csv",
            slate_key=slate_key,
        )
        if not blob_name:
            return None
        blob = self.bucket.blob(blob_name)
        return blob.download_as_text(encoding="utf-8")

    def write_lineup_version_csv(
        self,
        game_date: date,
        run_id: str,
        version_key: str,
        csv_text: str,
        slate_key: str | None = None,
    ) -> str:
        blob_name = self.lineup_version_csv_blob_name(game_date, run_id, version_key, slate_key=slate_key)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def read_lineup_version_upload_csv(
        self,
        game_date: date,
        run_id: str,
        version_key: str,
        slate_key: str | None = None,
    ) -> str | None:
        safe_version = self._safe_key(version_key, "version")
        blob_name = self._find_lineup_blob_name(
            game_date=game_date,
            run_id=run_id,
            relative_suffix=f"{safe_version}/dk_upload.csv",
            slate_key=slate_key,
        )
        if not blob_name:
            return None
        blob = self.bucket.blob(blob_name)
        return blob.download_as_text(encoding="utf-8")

    def write_lineup_version_upload_csv(
        self,
        game_date: date,
        run_id: str,
        version_key: str,
        csv_text: str,
        slate_key: str | None = None,
    ) -> str:
        blob_name = self.lineup_version_upload_blob_name(game_date, run_id, version_key, slate_key=slate_key)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(csv_text, content_type="text/csv")
        return blob_name

    def list_lineup_run_ids(self, game_date: date, slate_key: str | None = None) -> list[str]:
        prefix = f"{self.lineup_runs_prefix}/{game_date.isoformat()}/"
        blobs = self.bucket.list_blobs(prefix=prefix)
        target_slate = self._safe_slate_key(slate_key, default="main") if slate_key is not None else None
        run_ids: set[str] = set()
        for blob in blobs:
            name = str(blob.name or "")
            if not name.endswith("/manifest.json"):
                continue
            suffix = name[len(prefix) :]
            if not suffix:
                continue
            parts = [p for p in suffix.split("/") if p]
            if target_slate is None:
                if len(parts) >= 2 and parts[1] == "manifest.json":
                    run_ids.add(parts[0])
                    continue
                if len(parts) >= 3 and parts[2] == "manifest.json":
                    run_ids.add(parts[1])
                    continue
            else:
                if len(parts) >= 3 and parts[0] == target_slate and parts[2] == "manifest.json":
                    run_ids.add(parts[1])
                    continue
                if target_slate == "main" and len(parts) >= 2 and parts[1] == "manifest.json":
                    run_ids.add(parts[0])
        out = sorted(run_ids, reverse=True)
        return out

    def list_lineup_run_dates(self, slate_key: str | None = None) -> list[date]:
        prefix = f"{self.lineup_runs_prefix}/"
        blobs = self.bucket.list_blobs(prefix=prefix)
        target_slate = self._safe_slate_key(slate_key, default="main") if slate_key is not None else None
        dates_found: set[date] = set()
        for blob in blobs:
            name = str(blob.name or "")
            if not name.endswith("/manifest.json"):
                continue
            suffix = name[len(prefix) :]
            if not suffix:
                continue
            parts = [p for p in suffix.split("/") if p]
            if len(parts) < 3:
                continue
            date_part = parts[0].strip()
            if not date_part:
                continue
            if target_slate is not None:
                is_legacy_main = target_slate == "main" and len(parts) >= 3 and parts[2] == "manifest.json"
                is_target_slate = len(parts) >= 4 and parts[1] == target_slate and parts[3] == "manifest.json"
                if not (is_legacy_main or is_target_slate):
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
