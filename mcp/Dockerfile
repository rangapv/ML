# Runs a normal HTTP server (uvicorn) inside Lambda, using AWS's
# Lambda Web Adapter to translate Lambda invocations <-> HTTP requests.
# This lets the MCP Streamable HTTP transport work unmodified on Lambda.

FROM public.ecr.aws/docker/library/python:3.12-slim

# Lambda Web Adapter layer (as a multi-stage copy, no separate Lambda layer needed)
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8000
ENV AWS_LWA_INVOKE_MODE=response_stream

EXPOSE 8000

CMD ["python", "app.py"]
