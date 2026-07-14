# Company Data MCP Server (Lambda-ready)

Exposes your daily key-value dataset (fetched from an HTTPS endpoint) as MCP
tools that Claude (or any MCP client) can call: top-N by metric, field
coverage counts, filtering, and a free-form pandas fallback query.

## Files
- `app.py` — the MCP server (FastMCP) and all tool definitions
- `requirements.txt` — Python dependencies
- `Dockerfile` — packages the server for AWS Lambda via the Lambda Web Adapter

## Tools exposed
- `refresh_data()` — force re-fetch from the source endpoint now
- `list_columns()` — see available fields
- `get_top_n(metric, n=3, ascending=False)` — e.g. top 3 companies by value
- `get_field_coverage(field)` — e.g. how many companies have a ticker
- `filter_records(field, value, exact=True)` — e.g. all records where company="Acme"
- `run_query(pandas_expression)` — fallback for anything else, runs against `df`

## 1. Configure your data source

Set the environment variable before running/deploying:

```bash
export DATA_ENDPOINT_URL="https://your-endpoint.example.com/data"
```

The endpoint should return either:
- a JSON list: `[{"company": "Acme", "ticker": "ACM", "value": 1200000}, ...]`
- or a wrapper dict: `{"data": [...]}` (also handles "records"/"results"/"items")

## 2. Test locally first

```bash
pip install -r requirements.txt
export DATA_ENDPOINT_URL="https://your-endpoint.example.com/data"
python app.py
```

This starts a Streamable HTTP MCP server on `http://localhost:8000/mcp`.
Point Claude Desktop's remote MCP connector config (or any MCP Inspector
tool) at that URL to test the tools before deploying.

## 3. Deploy to Lambda (container image)

```bash
# Build
docker build -t company-data-mcp .

# Create an ECR repo (one-time)
aws ecr create-repository --repository-name company-data-mcp

# Authenticate, tag, and push
aws ecr get-login-password --region <region> | \
    docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

docker tag company-data-mcp:latest <account-id>.dkr.ecr.<region>.amazonaws.com/company-data-mcp:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/company-data-mcp:latest

# Create the Lambda function from the image
aws lambda create-function \
    --function-name company-data-mcp \
    --package-type Image \
    --code ImageUri=<account-id>.dkr.ecr.<region>.amazonaws.com/company-data-mcp:latest \
    --role arn:aws:iam::<account-id>:role/<your-lambda-execution-role> \
    --timeout 15 \
    --memory-size 512 \
    --environment "Variables={DATA_ENDPOINT_URL=https://your-endpoint.example.com/data,CACHE_TTL_SECONDS=86400}"
```

Notes:
- `--timeout 15`: covers a cold-start fetch + parse + query comfortably.
  Increase if your endpoint is slow.
- `--memory-size 512`: fine for small-to-medium datasets; bump up if you're
  loading a very large dataset into pandas.
- If your Lambda needs to be in a VPC for other reasons, make sure it has a
  NAT Gateway (or equivalent) for outbound internet access, or it won't be
  able to reach the HTTPS endpoint.

## 4. Expose it publicly (Function URL)

```bash
aws lambda create-function-url-config \
    --function-name company-data-mcp \
    --auth-type NONE \
    --invoke-mode RESPONSE_STREAM
```

This gives you a public HTTPS URL. **`--auth-type NONE` means anyone with the
URL can query your data** — fine for an internal team tool behind a hard-to-
guess URL, but if you need real access control, use `--auth-type AWS_IAM` and
have callers sign requests, or put API Gateway with an authorizer in front.

## 5. Connect from Claude

Add the Function URL (with `/mcp` path) as a remote MCP connector in Claude
Desktop / Claude Code / claude.ai, following Anthropic's remote MCP connector
setup instructions. Anyone with access to that connector can then ask
natural-language questions like:

- "What are the top 3 companies by value?"
- "How many companies do we have a ticker for?"
- "Show me all records where sector contains 'tech'"

Claude will pick the right tool, call it, and answer from the real
(computed, not guessed) result.

## Caching behavior

- Data is cached in memory per warm Lambda container for `CACHE_TTL_SECONDS`
  (default 24h).
- A cold start (or expired cache) triggers exactly one fetch from
  `DATA_ENDPOINT_URL` before answering the query.
- Call `refresh_data()` any time to force an immediate re-fetch, e.g. if you
  know the source just updated.

## When to add S3 caching instead

If you notice cold-start latency becoming a real problem (slow endpoint,
large payload, frequent cold starts), add an S3-backed cache layer between
the endpoint and Lambda: write a JSON snapshot to S3 whenever refreshed, and
have `_fetch_records()` read from S3 first. Not needed until you actually
observe this being slow.
