# Web App (Vercel Starter)

Next.js frontend for replacing Streamlit UI with a production-grade interface.

## Local Development

```bash
cd apps/web
cp .env.example .env.local
npm install
npm run dev
```

Default API base URL:

- `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`

## Implemented Pages

- `/` migration dashboard + registry coverage
- `/lineups` async lineup job submit/poll/download UI
- `/generate-lineup/game-data` slate readiness status surface
- `/generate-lineup/saved-runs` saved lineup run browser + version detail

## Deploy to Vercel

1. Create a new Vercel project from this repository.
2. Set Root Directory to `apps/web`.
3. Set environment variable `NEXT_PUBLIC_API_BASE_URL` to your API deployment URL.
4. Deploy.

## Architecture Plan

The concrete Streamlit-to-Vercel migration plan for this repo lives in:

- [docs/vercel_migration_plan.md](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/docs/vercel_migration_plan.md)
