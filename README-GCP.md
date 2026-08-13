# PPL Enterprise Intelligence — GCP Cloud Run + Vertex AI

Full-size local models (`bge-m3`, `bge-reranker-v2-m3`) on CPU, generation via Vertex AI (Gemini),
hosted on Cloud Run. This is the GCP deployment profile of the unified V8 codebase — set
`LLM_BACKEND=vertex` (auto-selected when `VERTEX_PROJECT_ID` is set).

For local Ollama setup, see [README.md](README.md). Architecture details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Why Cloud Run instead of a VM

Once generation moved to a managed API, there's no longer a reason to run a persistent VM —
nothing needs to stay resident except the embedding/reranker models, which get loaded once per
container instance and reused. Cloud Run auto-scales instances under load, has its own separate
Always Free monthly tier (on top of the $300 credit), and supports up to 32GB RAM / 8 vCPU per
instance — comfortably enough for the full-size models, unlike Render's 512MB ceiling.

## One-time setup

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com

# Grant the Cloud Run service account permission to call Vertex AI. Find your project number:
gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)'

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

No API key needed for Vertex — Cloud Run's attached service account handles authentication
automatically via Application Default Credentials. `VERTEX_PROJECT_ID`/`VERTEX_LOCATION` in the
deploy command below just tell the app which project/region to address.

## Deploy

```bash
gcloud run deploy ppl-enterprise-intelligence \
  --source . \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars LLM_BACKEND=vertex,VERTEX_PROJECT_ID=YOUR_PROJECT_ID,VERTEX_LOCATION=us-central1,VERTEX_MODEL=gemini-2.5-flash \
  --allow-unauthenticated
```

`--memory 4Gi --cpu 2` is a starting point, not a guarantee — `bge-m3` + `bge-reranker-v2-m3` in
fp32 plus PyTorch/FastAPI overhead is a real amount of RAM. Watch the build/deploy logs and the
revision's memory graph after first traffic (see checklist below); raise to `--memory 8Gi` if you
see it running close to the ceiling.

**If the build times out or fails during `RUN python -m app.ingest`** (full-size models + the full
205-page document is a heavier build than Render's ever was), use the included `cloudbuild.yaml`
for a bigger build machine instead:

```bash
gcloud builds submit --config cloudbuild.yaml
gcloud run deploy ppl-enterprise-intelligence \
  --image gcr.io/YOUR_PROJECT_ID/ppl-enterprise-intelligence \
  --region us-central1 --memory 4Gi --cpu 2 --timeout 300 \
  --set-env-vars LLM_BACKEND=vertex,VERTEX_PROJECT_ID=YOUR_PROJECT_ID,VERTEX_LOCATION=us-central1,VERTEX_MODEL=gemini-2.5-flash \
  --allow-unauthenticated
```

## Pre-flight validation checklist

Learned the hard way from the Render deploys — check these **in order**, don't skip to "does the
chat work":

1. **Watch the build log live.** `gcloud run deploy --source .` streams Cloud Build output to your
   terminal. Confirm you see `Extracted N evidence units from 1 PDF(s)` and `Index written to
   /app/data/index` before the build reports success — if it stalls or the process gets killed
   silently partway through, that's the OOM/timeout failure mode, and `cloudbuild.yaml` is the fix.
2. **Hit `/api/health` first**, before trying chat. It's instant (no model loading) and tells you
   immediately whether the index built (`"index": true`), which backend is active
   (`"llm_backend": "vertex"`), and whether Vertex is configured
   (`"vertex_project_configured": true`) — a fast way to isolate "index problem" from "Vertex
   problem" from "app didn't start at all."
3. **Send one `/api/chat` request and expect it to be slow the first time.** Models load lazily on
   first request (same fix as the Render build — an eager startup hook risks Cloud Run's own
   startup probe timing out). A multi-second delay on the very first query is normal, not a bug;
   subsequent queries on the same instance should be much faster.
4. **Check the revision's memory graph in Cloud Console** (Cloud Run → your service → Metrics)
   after a few requests. If it's consistently near your `--memory` limit, bump it — don't wait for
   an OOM to tell you.
5. **Confirm the IAM binding actually took effect** if `/api/chat` returns a Vertex-related error —
   the most common cause is the service account not having `roles/aiplatform.user` yet, or a typo
   in `VERTEX_PROJECT_ID`.

## Cost / credit monitoring

Cloud Run compute is billed against the $300 credit like any other covered product (not in the
documented exclusion list — only GPUs, Cloud Marketplace, quota increases, and Google AI Studio's
separate Gemini Developer API are excluded). Vertex AI/Gemini calls are also covered. Set a budget
alert so you're not surprised near the 90-day mark:

```bash
gcloud billing budgets create --billing-account=YOUR_BILLING_ACCOUNT_ID \
  --display-name="PPL demo budget" --budget-amount=250USD
```

## Corporate profile

For in-boundary deployments, set `DEPLOYMENT_PROFILE=corporate` along with `LLM_BACKEND=vertex`
(or `ollama` for fully self-hosted). External-only backends are rejected at startup.

## What I could not test

I have no GCP account access from this environment — everything above is syntax-checked and
logic-validated locally (same extraction pipeline already proven against the real 205-page
document), but not run end-to-end against live Cloud Run or Vertex AI. Treat the checklist above as
the way to catch problems in your own logs quickly, rather than assuming a clean first deploy.
