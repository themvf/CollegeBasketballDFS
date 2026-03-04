# API Service (Migration Starter)

This FastAPI service is the backend slice for moving the Streamlit app to a Vercel-first UI.

## Run Locally

```bash
pip install -r requirements.txt
pip install -r ../../requirements.txt
pip install -e ../..
uvicorn main:app --reload --port 8000
```

## Endpoints

- `GET /health`
- `GET /v1/rotowire/slates`
- `GET /v1/registry/coverage`
- `POST /v1/registry/import-dk-slate`

Pass your RotoWire member cookie as request header:

- `X-Rotowire-Cookie: <cookie>`

`POST /v1/registry/import-dk-slate` persists derived overrides to:

- `data/dk_manual_overrides.csv`

If your production deployment should not write local disk, set `persist=false` and store the response in your own persistence layer.
