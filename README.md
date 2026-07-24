# Techcombank FY Report Chatbot

A RAG (retrieval-augmented generation) chatbot that answers questions about
Techcombank's FY performance, grounded strictly in the official press
release PDF. Built with FastAPI, deployed as a containerized AWS Lambda
function via Terraform, provisioned through GitHub Actions.

## Architecture

```
PDF report --(ingest.py, run once)--> chunks + FAISS index (committed to repo)
                                              |
                                              v
User <--> frontend/index.html <--> FastAPI /chat <--> FAISS retrieval --> Bedrock Claude (answer, grounded in retrieved chunks only)
```

- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Generation:** Amazon Bedrock Claude 3 Sonnet
- **Vector store:** FAISS (flat index, in-process, no external DB needed —
  appropriate for a single ~20-page report)
- **Guardrail against hallucination:** the system prompt restricts the model
  to only the retrieved chunks, and explicitly instructs it to say "I don't
  have that information in the report I was given" rather than guess.
- **Multi-turn:** each `/chat` call takes a `session_id`; conversation
  history is kept in-memory server-side and passed back into the model on
  each turn so follow-ups ("what about the year before?") resolve correctly.

## Repo layout

```
app/            FastAPI service + RAG logic + ingest script
tests/          Unit tests for ingest.py/rag.py pure logic (pytest)
infra/          Terraform: ECR repo, IAM role, Lambda function, Function URL
.github/workflows/deploy.yml   CI/CD: build image -> push to ECR -> terraform apply
frontend/       Static single-page chat UI
Dockerfile       Production image (AWS Lambda container runtime)
Dockerfile.local Local dev image (plain uvicorn), used by docker-compose
docker-compose.yml   Local run
```

## Prerequisites

- Docker + Docker Compose
- An AWS account with Bedrock access enabled for Titan Embeddings and Claude
  3 Sonnet in your target region
- AWS credentials available locally (for running `ingest.py` and for local
  `docker-compose` runs)
- Terraform >= 1.5 (only needed for manual/local deploys — CI installs it
  automatically)

## 1. Generate the knowledge base (optional — only if regenerating the index)

**Skip this section for a normal demo/review** — `app/data/index.faiss` and
`app/data/chunks.json` are already committed to this repo. Only run this if
you're changing the source PDF, the chunking logic, or the embedding model
and need to rebuild the index from scratch.

Runs entirely inside Docker — no local Python/FAISS install needed:

```bash
git clone <this-repo>
cd techcombank-rag-chatbot
cp .env.example .env   # fill in your AWS credentials
set -a && source .env && set +a   # load them into your shell for docker compose

docker compose run --rm ingest --source "https://techcombank.com/content/dam/techcombank/public-site/documents/fy25-press-release-eng-12022026.pdf"
```

This writes `app/data/index.faiss` and `app/data/chunks.json` onto your host
(via a bind mount) — commit these two files, since they're what gets baked
into the production Docker image, so the container needs no network access
to the source PDF at runtime.

## 2. Run locally

The knowledge base (`app/data/index.faiss`, `app/data/chunks.json`) is
already committed to this repo — **no need to run `ingest` for a demo or
review**, only the API needs to be built and started:

```bash
podman-compose build api
podman-compose up -d api
```

API is now live at `http://localhost:8000`. Open `frontend/index.html`
directly in a browser (or `python -m http.server` inside `frontend/`) and
point the "API base URL" field at `http://localhost:8000`.

Quick smoke test:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What was Techcombank'\''s net profit for the fiscal year?"}'
```

**Measured timing** (build + start + first response, this machine, warm
Podman layer cache): ~1 min 28 sec — under the 3-minute target. Note this
reflects a machine with the base image layers already cached; a completely
cold machine (first-ever build, no cached layers) will take longer, since
most of that time is installing PyTorch/sentence-transformers/pdfplumber
from scratch rather than the app itself starting — realistically another
1-3 minutes depending on network speed. Only `ingest` needs AWS-independent
network access to download the source PDF; it isn't part of this timing
since it doesn't need to run for a normal demo/review.

## 3. Run unit tests

Unit tests cover the pure logic — table parsing/splitting, sentence-aware
chunking, page citation extraction, and citation snippet building — without
needing AWS credentials, network access, or a pre-built index:

```bash
pip install -r tests/requirements-test.txt
pytest
```

These also run automatically in CI before every deploy (see
`.github/workflows/deploy.yml`) — a failing test blocks the deploy job.

## 4. Test the PRODUCTION image locally (optional, before deploying)

`Dockerfile` (not `Dockerfile.local`) builds the actual image that goes to
Lambda — it uses AWS's Lambda base image and expects Lambda's invocation
protocol, not plain HTTP, so you can't `curl` it the normal way. AWS
provides a Runtime Interface Emulator (RIE) for exactly this:

```bash
# build the real production image
docker build -t rag-chatbot-prod .

# run it with the RIE (built into the AWS base image already)
docker run --rm -p 9000:8080 \
  -e AWS_REGION=us-east-1 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  rag-chatbot-prod

# in another terminal, invoke it Lambda-style:
curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{
        "version": "2.0",
        "routeKey": "POST /chat",
        "rawPath": "/chat",
        "requestContext": {"http": {"method": "POST", "path": "/chat"}},
        "headers": {"content-type": "application/json"},
        "body": "{\"message\": \"What was net profit for the fiscal year?\"}",
        "isBase64Encoded": false
      }'
```

This confirms the Mangum adapter + Lambda packaging works *before* you push
to AWS — much faster feedback loop than waiting on a GitHub Actions run.

## 5. Deploy to AWS

### One-time setup
1. Create a dedicated IAM user for CI (e.g. `github-actions-deploy`) with
   permissions for ECR, Lambda, IAM (to create/configure the Lambda's
   execution role), and Bedrock. (OIDC role-based auth was attempted first —
   simpler, no long-lived credentials — but hit an `AssumeRoleWithWebIdentity`
   authorization failure that persisted even with a verified-correct trust
   policy and provider config, most likely an account-level SCP restriction.
   Fell back to access keys to keep moving; worth revisiting with org-admin
   access if this were going to production.)
2. Generate an access key for that user and add two repo secrets:
   `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
3. (Optional but recommended) uncomment the S3 backend block in
   `infra/main.tf` and create that bucket, so Terraform state isn't lost
   between CI runs.

### Deploy
Push to `main`, or run the workflow manually from the Actions tab. The
pipeline:
1. Creates the ECR repository (first Terraform pass, since the repo must
   exist before we can push an image to it)
2. Builds and pushes the Docker image tagged with the commit SHA
3. Runs Terraform again to point the Lambda function at that new image and
   provision (or update) the IAM role and public Function URL
4. Prints the deployed API URL

## Design notes / trade-offs

- **Local embeddings instead of Bedrock Titan:** embedding runs locally
  inside the container via Sentence-Transformers rather than calling
  Bedrock per chunk. This removes an entire class of runtime dependency
  (API quota, network latency, throttling) from both ingestion and live
  query retrieval — only the final answer-generation call goes to Bedrock.
  Trade-off: the container image is larger (includes PyTorch + model
  weights) and cold starts are a bit slower, which is why Lambda memory
  was bumped to 2048MB.
- **Lambda + Function URL** instead of ECS/API Gateway: fewer moving parts
  to define in IaC, cold-start is acceptable for a Q&A chatbot, and it's the
  fastest path to a genuinely working deployed endpoint within the time box.
  Trade-off: less suited to sustained high-throughput traffic — ECS Fargate
  would be the next step if this needed to scale.
- **FAISS over a managed vector DB:** a single ~20-page report has a small
  enough chunk count that a flat in-memory index is both simpler and faster
  than standing up OpenSearch/Kendra, with no accuracy penalty at this
  scale.
- **In-memory session store:** acceptable for a demo/single-instance
  deployment; would move to DynamoDB if this needed to survive restarts or
  run behind multiple concurrent Lambda instances reliably.
- **Function URL auth is `NONE`** for demo convenience — for anything beyond
  a take-home, switch to `AWS_IAM` auth or put it behind API Gateway with a
  key.

## Known limitations

- Answers are only as good as what the PDF text extraction captures — if
  the report relies heavily on charts/images rather than text, those
  figures won't be retrievable. Worth spot-checking after ingestion.
- Single-region, single-environment setup (no staging/prod split) to keep
  the IaC footprint small for this exercise.
