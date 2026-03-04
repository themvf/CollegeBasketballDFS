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

## Deploy to Vercel

1. Create a new Vercel project from this repository.
2. Set Root Directory to `apps/web`.
3. Set environment variable `NEXT_PUBLIC_API_BASE_URL` to your API deployment URL.
4. Deploy.
