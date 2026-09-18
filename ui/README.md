# gbench Web UI

The gbench dashboard: a React + TypeScript + Vite single-page app (Material UI (MUI),
Recharts) that talks to the FastAPI service in [`../service/`](../service/) over its
`/api` REST endpoints. It lets you submit benchmark/eval runs, stream their logs
live, and compare results across models and runs.

## Pages (`src/pages/`)
- **Dashboard** (`Dashboard.tsx`, `/`) - active job monitors, recent-runs feed, quick
  metric cards (TTFT, TPOT, throughput), and system/hardware status.
- **New Evaluation** (`NewEvaluation.tsx`, `/new`) - configure and launch a run:
  model, endpoint, performance campaign, batch sizes, native evals, Golden Set, or
  GemmaClaw scenarios.
- **Compare** (`Compare.tsx`, `/compare`) - multi-model / multi-run charts for
  P50/P99 TTFT, TPOT, output-token throughput, and pass-rate radars.
- **Analytics** (`Analytics.tsx`, `/analytics`) - latency distributions, Coefficient
  of Variation (CV%) tables, and raw sample-trace logs.

## Running locally
Start the backend first (from the repo root), then the Vite dev server:

```bash
# 1. Backend API (repo root) - serves on http://localhost:8000
python -m service.main

# 2. Frontend (this directory) - serves on http://localhost:5173, proxies /api -> :8000
cd ui
npm install
npm run dev
```

See the repository [README](../README.md#web-ui-dashboard--rest-api-service) for the
full dashboard + REST API reference, and the API endpoint table there for what the
frontend consumes.

## Tooling
Vite (`vite.config.ts`) with the React plugin, TypeScript, Material UI (MUI), and Oxlint.
Lint with `npm run lint`; build a production bundle with `npm run build`.
