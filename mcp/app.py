"""
MCP server exposing your daily key-value dataset as queryable tools.

Runs as a normal HTTP server locally (for testing) and inside AWS Lambda
via the Lambda Web Adapter (for deployment) using MCP's Streamable HTTP
transport, so it works as a remote MCP connector for you and others.

Data source: an HTTPS endpoint that returns JSON (list of key-value records).
Caching: in-memory, refreshed only when older than CACHE_TTL_SECONDS.
"""
import asyncio
import os
import sys
import time
import json
import threading
from typing import Any

import httpx
import pandas as pd
from mcp.server.fastmcp import FastMCP
from fastmcp import Client

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
 
 
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}/mcp"
 



# --------------------------------------------------------------------------
# Configuration (set these as Lambda environment variables in production)
# --------------------------------------------------------------------------
DATA_ENDPOINT_URL = os.environ.get("DATA_ENDPOINT_URL", "https://ncrwopyvdqpnov6hf27omijv6e0ekmuo.lambda-url.us-west-2.on.aws/")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", str(24 * 60 * 60)))  # 24h
FETCH_TIMEOUT_SECONDS = float(os.environ.get("FETCH_TIMEOUT_SECONDS", "700"))

# --------------------------------------------------------------------------
# In-memory cache. Survives across invocations only on "warm" Lambda
# containers -- that's fine here, since a cold start just re-fetches once.
# --------------------------------------------------------------------------
_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"df": None, "fetched_at": 0.0}


def _fetch_records() -> list[dict]:
    """Fetch raw key-value records from the HTTPS endpoint."""
    resp = httpx.get(DATA_ENDPOINT_URL, timeout=FETCH_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()

    # Be tolerant of either a bare list, or a wrapper like {"data": [...]}
    if isinstance(data, dict):
        for key in ("data", "records", "results", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(
            "Expected the endpoint to return a JSON list of records "
            f"(or a dict wrapping one), got: {type(data)}"
        )
    return data


def get_dataframe(force_refresh: bool = False) -> pd.DataFrame:
    """Return a cached DataFrame, refreshing from the endpoint if stale."""
    with _cache_lock:
        age = time.time() - _cache["fetched_at"]
        is_stale = _cache["df"] is None or age > CACHE_TTL_SECONDS

        if is_stale or force_refresh:
            records = _fetch_records()
            _cache["df"] = pd.DataFrame(records)
            _cache["fetched_at"] = time.time()

        return _cache["df"]


# --------------------------------------------------------------------------
# MCP server + tools
# --------------------------------------------------------------------------
mcp = FastMCP("company-data" , host=HOST, port=PORT)

client = Client(mcp)

@mcp.tool()
def refresh_data() -> str:
    """Force a re-fetch of the dataset from the source endpoint right now,
    instead of waiting for the normal 24h cache to expire."""
    df = get_dataframe(force_refresh=True)
    return f"Refreshed. {len(df)} records loaded. Columns: {list(df.columns)}"


@mcp.tool()
def list_columns() -> str:
    """List the available column/field names in the dataset, so you know
    what you can filter, group, or rank by."""
    df = get_dataframe()
    return json.dumps(list(df.columns))


@mcp.tool()
def get_top_n(metric: str, n: int = 3, ascending: bool = False) -> str:
    """Get the top N (or bottom N) records ranked by a numeric metric column.

    Args:
        metric: the column name to rank by, e.g. "value".
        n: how many records to return (default 3).
        ascending: if True, returns the smallest N instead of largest N.
    """
    df = get_dataframe()
    if metric not in df.columns:
        return f"Error: '{metric}' is not a column. Available: {list(df.columns)}"

    sorted_df = df.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
    result = sorted_df.head(n)
    return result.to_json(orient="records")


@mcp.tool()
def get_field_coverage(field: str) -> str:
    """Count how many records have a non-null value for a given field,
    and how many are missing it. Useful for questions like
    'how many companies do we have a ticker for'.

    Args:
        field: the column name to check, e.g. "ticker".
    """
    df = get_dataframe()
    if field not in df.columns:
        return f"Error: '{field}' is not a column. Available: {list(df.columns)}"

    total = len(df)
    present = int(df[field].notna().sum())
    missing = total - present
    return json.dumps(
        {"field": field, "total_records": total, "present": present, "missing": missing}
    )


@mcp.tool()
def filter_records(field: str, value: str, exact: bool = True) -> str:
    """Filter records where a given field matches a value.

    Args:
        field: the column name to filter on, e.g. "company".
        value: the value to match.
        exact: if True, requires an exact match; if False, does a
            case-insensitive substring match.
    """
    df = get_dataframe()
    if field not in df.columns:
        return f"Error: '{field}' is not a column. Available: {list(df.columns)}"

    if exact:
        result = df[df[field] == value]
    else:
        result = df[df[field].astype(str).str.contains(value, case=False, na=False)]

    return result.to_json(orient="records")


@mcp.tool()
def count_tickers() -> int:
    """Return the number of unique tickers."""
    df = get_dataframe()
    print(f'the df is {df}')
    return int(df['ticker'].nunique())


@mcp.tool()
def run_query(pandas_expression: str) -> str:
    """Fallback for anything not covered by the other tools. Runs a pandas
    expression against the dataset, where the DataFrame is available as `df`.
    Only read-only expressions are allowed (no assignment, no imports).

    Example expressions:
        "df.nlargest(3, 'value')[['company', 'value']]"
        "df['ticker'].notna().sum()"
        "df.groupby('sector')['value'].sum().sort_values(ascending=False)"

    Args:
        pandas_expression: a single Python expression referencing `df`.
    """
    df = get_dataframe()

    forbidden = ["import", "__", "exec", "eval", "open(", "os.", "sys.", "subprocess"]
    if any(tok in pandas_expression for tok in forbidden):
        return "Error: expression contains disallowed tokens."

    try:
        result = eval(pandas_expression, {"__builtins__": {}}, {"df": df, "pd": pd})
    except Exception as e:
        return f"Error running expression: {e}"

    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.to_json(orient="records" if isinstance(result, pd.DataFrame) else "table")
    return json.dumps(result, default=str)

def run_server():
    print(f"Starting MCP server at {URL} (Ctrl+C to stop)")
    mcp.run(transport="streamable-http")

async def run_client():
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            #print(f"Server: {client.initialize_result.serverInfo.name}")
 
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
           
            result = await count_tickers()
            print(f'result of count ticker is {result}')

            #result = await session.call_tool("run_query", {"text": "How many tickers are there"})
            #print("run_query('hello mcp') ->", result.content[0].text)
 



if __name__ == "__main__":
    # Local testing: runs a Streamable HTTP server on port 8000.
    # Point a local MCP client (or Claude Desktop's remote connector config)
    # at http://localhost:8000/mcp
    #mcp.run(transport="streamable-http")
    if "--server" in sys.argv:
        run_server()
    else:
        asyncio.run(run_client())
