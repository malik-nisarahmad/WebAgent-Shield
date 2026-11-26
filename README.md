# WebAgent Shield

AI-powered prompt injection defense for web-browsing agents.

Phases 1 and 2 implement a task-driven research agent and automatic HTML extraction. The agent searches using Tavily, chooses pages through an LLM with an OpenAI-compatible chat-completions endpoint, opens them in Playwright, and receives structured evidence with source locations.

## Local setup

Install Python 3.11 or newer, then run in PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python -m playwright install chromium
```

Create a local `.env` file and fill in the API credentials, model name, and comma-separated exact hostnames. Credentials stay on the server and `.env` is ignored by Git. Include the websites needed for your tasks and any necessary asset hosts. Then start the local API.

```powershell
.\.venv\Scripts\python -m uvicorn webagent_shield.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs and submit a task to `POST /v1/agent-runs`. The request waits for completion, up to the run's 180-second budget. The response includes the answer, opened source URLs, scans, and action records. Retrieve saved results through `GET /v1/agent-runs/{run_id}`. Local SQLite records live in `artifacts/runs.db`.

For Groq, use `LLM_BASE_URL=https://api.groq.com/openai/v1` and a tool-calling model available to your account, such as `LLM_MODEL=openai/gpt-oss-20b`. Set `LLM_API_KEY` and `TAVILY_API_KEY` locally. Restart the server after editing configuration. Model availability and account limits can change. Requests retry HTTP 429 responses up to twice within the overall run timeout. Page evidence sent to the LLM has a 2,000-character text budget per page, with truncation recorded in coverage metadata. This can omit useful information on long pages. Full extracted passages remain available in saved scans.

Suggested tasks include comparing two documented Python features, locating a university policy across approved pages, and summarizing two approved product specification pages. Configure the matching exact hostnames before each experiment.

## Current behavior and limits

- Both search results and page evidence pass through an inspection interface. ML detection and fine-tuning are scheduled for phase 3. Every result currently reports `not_trained`, and empty findings do not indicate safety.
- Raw HTML and a rendered snapshot are extracted automatically. Comments, hidden elements, and selected attributes are recorded for inspection. Only eligible rendered text is returned to the agent.
- HTML nodes and attributes have structural locations. Text offsets refer to normalized passages. Original byte offsets, cross-view deduplication, and closed shadow DOM support remain future extraction improvements.
- Rendering captures the main document after a bounded delay. Dynamic content arriving later and iframe content may be absent. Blocked requests and passage truncation produce partial-coverage flags.
- The gateway permits at most three searches and five page opens. Browser requests are restricted to configured public hosts and GET requests. Service workers and WebSockets are blocked. Responses and rendered documents have size limits.
- DNS validation precedes transport resolution. Deployments need network-level egress enforcement to close DNS-rebinding races. This local prototype should use operator-selected trusted public hosts and an isolated environment when inspecting adversarial pages.
- Saved task and evidence records can contain sensitive content. The local API is unauthenticated and intended for loopback use. Redaction and multi-user access control are pending.
- Final citation correctness requires evaluation. The API separately records actual opened source URLs for comparison with the answer.

## Verification

```powershell
.\.venv\Scripts\python -m pytest
```

Tests cover hidden-content handling, source provenance, destination scope, private-address rejection, and denial of unknown tools. Live acceptance requires configured API keys, Chromium, and an allowed public website. Run a task that needs two sources, verify both appear in `scans`, and verify the answer cites the opened sources. If Python is not installed, install Python 3.11 or newer before creating `.venv`.
