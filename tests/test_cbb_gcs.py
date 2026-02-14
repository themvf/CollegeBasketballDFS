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
