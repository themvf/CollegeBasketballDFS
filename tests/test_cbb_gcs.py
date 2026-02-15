from datetime import date

from college_basketball_dfs.cbb_gcs import CbbGcsStore


class _FakeBlob:
    def __init__(self, name: str, store: dict[str, str]) -> None:
        self.name = name
        self._store = store

    def exists(self) -> bool:
        return self.name in self._store

    def upload_from_string(self, payload: str, content_type: str | None = None) -> None:
        self._store[self.name] = payload

    def download_as_text(self, encoding: str = "utf-8") -> str:
        return self._store[self.name]

    def delete(self) -> None:
        self._store.pop(self.name, None)


class _FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, str] = {}

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(name, self.objects)

    def list_blobs(self, prefix: str):
        for key in sorted(self.objects):
            if key.startswith(prefix):
                yield _FakeBlob(key, self.objects)


class _FakeClient:
    def __init__(self) -> None:
        self._bucket = _FakeBucket()

    def bucket(self, name: str) -> _FakeBucket:
        return self._bucket


def test_dk_slate_blob_name_and_rw() -> None:
    store = CbbGcsStore(bucket_name="bucket", client=_FakeClient())
    d = date(2026, 2, 14)
    expected_blob = "cbb/dk_slates/2026-02-14_dk_slate.csv"

    assert store.dk_slate_blob_name(d) == expected_blob
    written = store.write_dk_slate_csv(d, "col1,col2\n1,2\n")
    assert written == expected_blob
    assert store.read_dk_slate_csv(d) == "col1,col2\n1,2\n"
    assert store.delete_dk_slate_csv(d) is True
    assert store.read_dk_slate_csv(d) is None
    assert store.delete_dk_slate_csv(d) is False


def test_dk_slate_read_missing_returns_none() -> None:
    store = CbbGcsStore(bucket_name="bucket", client=_FakeClient())
    assert store.read_dk_slate_csv(date(2026, 2, 15)) is None


def test_injuries_blob_name_and_rw() -> None:
    store = CbbGcsStore(bucket_name="bucket", client=_FakeClient())
    expected_blob = "cbb/injuries/injuries_master.csv"
    assert store.injuries_blob_name() == expected_blob
    written = store.write_injuries_csv("player_name,team,status,active\nA,AAA,Out,true\n")
    assert written == expected_blob
    assert "A,AAA,Out,true" in (store.read_injuries_csv() or "")


def test_projections_blob_name_and_rw() -> None:
    store = CbbGcsStore(bucket_name="bucket", client=_FakeClient())
    d = date(2026, 2, 14)
    expected_blob = "cbb/projections/2026-02-14_projections.csv"
    assert store.projections_blob_name(d) == expected_blob
    written = store.write_projections_csv(d, "id,proj\n1,25.5\n")
    assert written == expected_blob
    assert "25.5" in (store.read_projections_csv(d) or "")


def test_ownership_blob_name_and_rw() -> None:
    store = CbbGcsStore(bucket_name="bucket", client=_FakeClient())
    d = date(2026, 2, 14)
    expected_blob = "cbb/ownership/2026-02-14_ownership.csv"
    assert store.ownership_blob_name(d) == expected_blob
    written = store.write_ownership_csv(d, "id,actual_ownership\n1,12.5\n")
    assert written == expected_blob
    assert "12.5" in (store.read_ownership_csv(d) or "")


def test_contest_standings_blob_name_and_rw() -> None:
    store = CbbGcsStore(bucket_name="bucket", client=_FakeClient())
    d = date(2026, 2, 14)
    expected_blob = "cbb/contest_standings/2026-02-14_contest-123.csv"
    assert store.contest_standings_blob_name(d, "contest-123") == expected_blob
    written = store.write_contest_standings_csv(d, "contest-123", "Rank,EntryId\n1,123\n")
    assert written == expected_blob
    assert "EntryId" in (store.read_contest_standings_csv(d, "contest-123") or "")


def test_lineup_runs_blob_names_and_rw() -> None:
    store = CbbGcsStore(bucket_name="bucket", client=_FakeClient())
    d = date(2026, 2, 14)
    run_id = "run_abc123"
    version_key = "spike_v1"

    assert store.lineup_run_prefix(d, run_id) == "cbb/lineup_runs/2026-02-14/run_abc123"
    assert store.lineup_run_manifest_blob_name(d, run_id) == "cbb/lineup_runs/2026-02-14/run_abc123/manifest.json"
    assert (
        store.lineup_version_json_blob_name(d, run_id, version_key)
        == "cbb/lineup_runs/2026-02-14/run_abc123/spike_v1/lineups.json"
    )
    assert (
        store.lineup_version_csv_blob_name(d, run_id, version_key)
        == "cbb/lineup_runs/2026-02-14/run_abc123/spike_v1/lineups.csv"
    )
    assert (
        store.lineup_version_upload_blob_name(d, run_id, version_key)
        == "cbb/lineup_runs/2026-02-14/run_abc123/spike_v1/dk_upload.csv"
    )

    manifest_payload = {"run_id": run_id, "slate_date": d.isoformat()}
    manifest_blob = store.write_lineup_run_manifest_json(d, run_id, manifest_payload)
    assert manifest_blob.endswith("/manifest.json")
    assert store.read_lineup_run_manifest_json(d, run_id) == manifest_payload

    lineups_payload = {"lineup_count": 2, "lineups": [{"lineup_number": 1}, {"lineup_number": 2}]}
    json_blob = store.write_lineup_version_json(d, run_id, version_key, lineups_payload)
    assert json_blob.endswith("/lineups.json")
    assert store.read_lineup_version_json(d, run_id, version_key) == lineups_payload

    csv_blob = store.write_lineup_version_csv(d, run_id, version_key, "Lineup,Salary\n1,50000\n")
    upload_blob = store.write_lineup_version_upload_csv(d, run_id, version_key, "G,G,G,F,F,F,UTIL,UTIL\n")
    assert csv_blob.endswith("/lineups.csv")
    assert upload_blob.endswith("/dk_upload.csv")
    assert "Salary" in (store.read_lineup_version_csv(d, run_id, version_key) or "")
    assert "UTIL" in (store.read_lineup_version_upload_csv(d, run_id, version_key) or "")


def test_list_lineup_run_ids() -> None:
    store = CbbGcsStore(bucket_name="bucket", client=_FakeClient())
    d = date(2026, 2, 14)
    store.write_lineup_run_manifest_json(d, "run_a", {"run_id": "run_a"})
    store.write_lineup_run_manifest_json(d, "run_b", {"run_id": "run_b"})
    store.write_lineup_run_manifest_json(date(2026, 2, 15), "run_other_day", {"run_id": "run_other_day"})

    run_ids = store.list_lineup_run_ids(d)
    assert run_ids == ["run_b", "run_a"]


def test_list_lineup_run_dates() -> None:
    store = CbbGcsStore(bucket_name="bucket", client=_FakeClient())
    store.write_lineup_run_manifest_json(date(2026, 2, 14), "run_a", {"run_id": "run_a"})
    store.write_lineup_run_manifest_json(date(2026, 2, 16), "run_b", {"run_id": "run_b"})
    store.write_lineup_run_manifest_json(date(2026, 2, 15), "run_c", {"run_id": "run_c"})

    dates = store.list_lineup_run_dates()
    assert dates == [date(2026, 2, 16), date(2026, 2, 15), date(2026, 2, 14)]


def test_phantom_review_blob_names_and_rw() -> None:
    store = CbbGcsStore(bucket_name="bucket", client=_FakeClient())
    d = date(2026, 2, 14)
    run_id = "run_xyz"
    version_key = "spike_v2_tail"

    expected_csv_blob = "cbb/phantom_reviews/2026-02-14/run_xyz/spike_v2_tail.csv"
    expected_summary_blob = "cbb/phantom_reviews/2026-02-14/run_xyz/summary.json"
    assert store.phantom_review_csv_blob_name(d, run_id, version_key) == expected_csv_blob
    assert store.phantom_review_summary_blob_name(d, run_id) == expected_summary_blob

    written_csv = store.write_phantom_review_csv(d, run_id, version_key, "lineup_number,actual_points\n1,210.5\n")
    assert written_csv == expected_csv_blob
    assert "210.5" in (store.read_phantom_review_csv(d, run_id, version_key) or "")

    summary_payload = {"run_id": run_id, "models": 1}
    written_summary = store.write_phantom_review_summary_json(d, run_id, summary_payload)
    assert written_summary == expected_summary_blob
    assert store.read_phantom_review_summary_json(d, run_id) == summary_payload
