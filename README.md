# maritime-safety-copilot

A small, local Python RAG demo for exploring maritime safety study notes. Ask a
question in Streamlit, get an answer from FastAPI, and inspect the retrieved
excerpts and scores. Built as a one-day Academy MVP.

**Educational demo, not a regulatory authority or operational safety tool.** The
knowledge base contains curated educational summaries and study notes, not
full copyrighted regulatory publications. It cannot establish compliance, current
legal requirements, vessel applicability, or emergency procedures.

## Architecture

```text
Streamlit (127.0.0.1:8501)
  session history + controls + evidence display
           |
           | POST /ask/stream over local HTTP (incremental NDJSON)
           v
FastAPI (127.0.0.1:8000)
  validate -> route query -> retrieve -> build prompt -> Azure OpenAI chat
                               |
                    local data/*.md and *.txt
                    keywords + in-memory embeddings/cosine
           |
           +-- optional Langfuse observations at an explicitly configured URL
```

The frontend imports no backend or model code. The backend stores no conversations.
Knowledge files stay on the local filesystem; document embeddings are cached only
in backend memory. Azure OpenAI is used only for embedding and chat API calls.
No Azure deployment, Blob Storage, Cosmos DB, Functions, vector database, Docker,
authentication service, workers, or other application persistence is involved.

```text
backend/main.py       API contracts, /health, /ask and request orchestration
backend/retrieval.py  chunking, routing, keyword scoring and cosine similarity
backend/llm.py        environment loading, Azure client and embedding cache
backend/prompts.py    grounded/general prompts and insufficient-evidence response
backend/tracing.py    optional, best-effort Langfuse observations
frontend/app.py       Streamlit chat, prompt buttons, uploader, sources and session history
frontend/uploads.py   safe local Markdown upload saving
frontend/streaming.py incremental HTTP event reader for Streamlit
assets/              maritime hero banner image
data/                dynamically discovered Markdown/text files with optional YAML metadata
tests/               offline retrieval, API, provider, tracing and UI tests
```

## Local setup

Requires **Python 3.12+**. Run all commands from this directory (`Day18` in the
Academy repository, or the root after copying it into a standalone repository).
Use a chat deployment supporting Chat Completions with `max_completion_tokens`
and a separate text-embedding deployment in an existing Azure OpenAI resource.

Chat calls allow 4,096 completion tokens, including reasoning tokens where applicable.
Prompts target answers of about 200 words unless more detail is requested. A response
with `finish_reason="length"` is incomplete and returns an actionable HTTP 502;
partial answers are not presented as complete. The budget provides headroom rather
than a guarantee for every model or question.

Windows PowerShell:

```powershell
cd Day18
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cd Day18
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

If already inside the project, omit `cd Day18`. If PowerShell activation is
restricted, invoke `.\.venv\Scripts\python.exe` instead of `python` in commands.

Edit `.env` with your own values:

| Variable | Purpose |
| --- | --- |
| `AZURE_OPENAI_ENDPOINT` | HTTPS resource endpoint |
| `AZURE_OPENAI_API_KEY` | Resource API key |
| `AZURE_OPENAI_API_VERSION` | API version supported by your resource and deployments |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Your chat **deployment name**, not a hard-coded model ID |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Your embedding deployment name |
| `BACKEND_URL` | Defaults to `http://127.0.0.1:8000` |

Only this project's `.env` is loaded; the parent Academy `.env` is not searched.
Existing process environment values take precedence. Restart both processes after
changing configuration. Never commit `.env` or credentials. The checked-in
`.env.example` contains placeholders only.

Start the backend in terminal 1:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in terminal 2, from the same directory and virtual environment:

```bash
python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8501**. API docs are at **http://127.0.0.1:8000/docs**;
`GET /health` checks process liveness without making a provider call. Both apps
start without Azure credentials. Generating answers needs chat credentials;
semantic/hybrid search also needs an embedding deployment. Missing configuration
returns an actionable HTTP 503; provider failures return a sanitized HTTP 502.

## Try the demo

Use the **Try a prompt** buttons beside the chat input for SOLAS IX, emergency
preparedness, or permits to work. A click immediately submits the full question
through the same flow as typed chat, using the current mode, controls and history.

The sidebar's optional **Add a knowledge file** section accepts UTF-8 `.md` files
up to 1 MB. Select a file and click **Save to knowledge base**. Files are saved into
this project's existing `data/` directory with a sanitized, unique filename;
existing files are preserved. YAML front matter is retained for the backend's
existing loader to parse. PDF and other formats are not supported.

The backend discovers uploaded files on the next retrieval; embeddings refresh on
the next semantic/hybrid retrieval. Restarting the backend also clears its cache,
but uploads do not require a restart. General LLM mode continues to bypass the KB.
The frontend and backend must use the same local project/data directory. Uploads
remain on disk after clearing chat; remove unwanted files from `data/` manually.
Only add curated material you have permission to use. Markdown/YAML parsing errors
are reported by the existing backend loader when the file is retrieved.

| Question | Expected route |
| --- | --- |
| What is SOLAS IX? | Exact identifier/keyword search |
| Explain ISM Code section 8. | Exact search |
| What is section 10 of the ISM Code about? | Exact search |
| How does ISM Code section 8 support emergency preparedness? | Hybrid search |
| Compare ISM Code sections 8 and 9. | Hybrid search |
| How can a crew learn from a near miss? | Semantic search |
| What does SOLAS XII require? | Exact search, no evidence in this KB |

In **Knowledge Base** mode the prompt permits only retrieved evidence, requires
`[S1]`-style citations, and asks the model to acknowledge unsupported parts. A
basic source-label check rejects missing or unknown citations. It checks labels,
not whether a claim is true. With no sources, the API returns a deterministic
insufficient-evidence answer without calling the chat model. Turning retrieval
off has the same result, preserving the meaning of grounded mode.

**General LLM** mode bypasses retrieval, clearly labels the answer as unverified,
and returns no local sources. Each mode has a separate short conversation in
Streamlit session state: six exchanges displayed, at most four sent to the model.
Clear chat clears both modes. History is not treated as regulatory evidence.
Source labels restart for every answer. Retrieval uses the latest question;
repeat the subject or identifier in follow-ups such as “explain section 8 again.”

Expand sources to read the exact excerpt supplied to the model. Technical details
show the route, retrieved chunk count, latency, rank scores, cosine similarities,
and keyword scores. Similarity is not an answer confidence or compliance score.

An API request using only the standard library:

```python
import json
from urllib.request import Request, urlopen

payload = {"question": "What is SOLAS IX?", "mode": "knowledge_base",
           "retrieval_enabled": True, "top_k": 3, "history": []}
request = Request("http://127.0.0.1:8000/ask", data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=120) as response:
    print(json.load(response))
```

## Streaming answers

The frontend calls `POST /ask/stream` and renders Azure text fragments with
Streamlit's `st.write_stream`. Both answer modes stream. Knowledge Base mode first
finishes the existing retrieval step and constructs the same grounded prompt;
General LLM mode bypasses retrieval as before. The Azure Chat Completions call uses
`stream=True` with the same 4,096-token completion budget. Retrieval, model startup,
and any model reasoning still occur before the first visible text fragment.

The endpoint accepts the same JSON body as `/ask` and returns
`application/x-ndjson`: one JSON object per line, progressively flushed over HTTP.

```json
{"type": "delta", "text": "A fragment of the answer"}
{"type": "done", "response": {"answer": "The checked complete answer", "mode": "knowledge_base", "sources": [], "technical_details": {"retrieval_method": "exact", "chunks_retrieved": 0, "latency_ms": 1234}, "warnings": []}}
```

The `done.response` has the existing `/ask` response schema, including full source
metadata, excerpts and scores. Its answer is authoritative. While streaming, text
is provisional: once generation ends, the same citation-label check runs. If it
fails, the frontend replaces provisional text with the deterministic
insufficient-evidence response. Sources and technical details appear after completion.
If Knowledge Base mode has no evidence or retrieval is off, the stream contains
only a `done` event with the existing deterministic response; no LLM call is made.

Provider failures, missing configuration, filtering, token limits and premature
upstream termination produce an `error` event, for example:

```json
{"type": "error", "status_code": 502, "detail": "Azure chat stream failed or was interrupted. Please retry."}
```

HTTP headers have already been sent, so stream errors carry their status in the
event rather than changing HTTP 200. Invalid request bodies still return HTTP 422.
The frontend clears incomplete text, shows the error and does not save a partial
exchange in history. A connection that ends without `done` is also treated as an
interruption. Disconnecting closes the active Azure stream. There are no automatic
generation retries or duplicate submissions.

Langfuse keeps the root request and LLM observation open throughout streaming and
records the collected complete answer, prompt, retrieval output and timings. Failed
or cancelled generations are marked incomplete; any recorded text is labeled partial.
Tracing remains best-effort. `POST /ask` remains available with its original
non-streaming JSON response and HTTP error behavior. Both endpoints share retrieval,
prompt construction and final answer checks.

## Retrieval mechanics

1. Discover all `.md` files directly in `data/` on every request (case-insensitive;
   `.txt` remains supported). README files are excluded by case-insensitive filename
   stem; other documentation can opt out with YAML `document_type: documentation`.
   Knowledge filenames are not hard-coded. Parse optional leading YAML front matter
   with `yaml.safe_load`, then split only the body at headings and at 180 words.
   Every chunk retains its actual filename as `source`, resolved `title`, `section`,
   `chunk_id`, and the full front-matter mapping as `metadata`. Title priority is
   YAML `title`, first Markdown H1, then filename stem. Invalid front matter returns
   an actionable HTTP 503 naming the file instead of silently dropping documents.
2. Normalize common names and identifiers (for example `SOLAS chapter 9` →
   `solas ix`). An identifier or known Code name routes to exact search. Words
   such as “how,” “compare,” or “apply” make that a hybrid query. Other queries use
   semantic search. IGF Code names, SOLAS regulation identifiers and common IMO
   resolution identifiers such as `MSC.581(110)` are also recognized. The router
   is deterministic, with no routing LLM call.
3. Exact search uses query-token overlap and identifier/name boosts. Filename,
   title, body, and YAML fields (including `keywords` and `reference_basis`) are
   searchable. Token overlap weights title/section/body matches at 70% and matches
   anywhere in the chunk plus metadata at 30%, so shared document keywords do not
   entirely drown out relevant sections. An explicit
   identifier must actually occur in a chunk; section 1 does not match section 10
   or 1.2. Specific subsections absent from the notes return no exact result.
4. Semantic search embeds the corpus and question with the same deployment and
   computes `dot(q, d) / (norm(q) * norm(d))` in NumPy. Zero vectors score zero.
   Corpus embeddings are reused in a small memory cache keyed by contents and
   embedding configuration. Embedding inputs include filename, title, section,
   body, and consistently serialized metadata. Query embeddings are fresh each request.
5. Hybrid search includes exact matches or semantic matches and ranks with
   `0.6 * max(cosine, 0) + 0.4 * keyword_score`. Exact matches can survive a low
   cosine score. Other semantic candidates need the similarity threshold (default
   `0.30`, configurable in the API as `min_similarity`). Exact routes make no
   embedding calls. Return up to `top_k` sources (1–6, default 3).

The threshold and hybrid weights are demo heuristics, not calibrated performance
claims. Hybrid results can include related notes even when a requested identifier
is absent; the prompt must still acknowledge insufficient evidence. Editing a
knowledge file's body, metadata, or filename changes the cache key on the next
request. Added and removed documents are reflected automatically. Nothing is
indexed on disk, so there is no rebuild command: the next semantic/hybrid request
generates or reuses embeddings for exactly the current corpus. Restart the backend
after changing Python code; content edits alone need no restart.

YAML front matter is kept out of body excerpts and exposed separately in the source
viewer and API. Nested mappings, lists, strings, numbers, booleans and null values
are preserved; YAML dates/timestamps become ISO strings for JSON transport. Metadata
identifies and describes a source; a listed reference does not mean the original
regulatory publication was retrieved. See [data/README.md](data/README.md) for the
document format. Metadata-only files produce no evidence chunks.

## Optional Langfuse

The app works with no Langfuse package or configuration. To enable it:

```bash
python -m pip install -r requirements-langfuse.txt
```

Set **all three** variables in `.env`: `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`, then restart the backend. There is
no implicit hosted URL. Point at an existing local Langfuse instance (for example
`http://localhost:3000`) to keep traces local. Setting up a Langfuse server is
outside this MVP; no infrastructure is created here.

The root observation records the question, mode, history, final answer and total
latency. Child observations record retrieval inputs/results/scores and the exact
prompt and LLM response, with step timing. Missing/partial config disables
tracing; SDK initialization, update and export failures do not fail an answer.
The SDK flushes on backend shutdown. Trace delivery still depends on a reachable,
correctly configured instance.

**Data flow:** Azure receives knowledge text for embeddings and the current
prompt/history for chat. Enabling Langfuse also sends questions, excerpts, prompts,
history and answers to the configured instance, where traces may be retained.
Leave Langfuse unset for no telemetry export, or use a local instance if cloud
trace persistence is unwanted. No application data is written to a cloud store.

## Tests and Academy concepts

```bash
python -m pytest -q
```

Tests run offline with fake embedding vectors/model responses. They cover routing,
identifier boundaries, YAML parsing/metadata preservation, current curated-document
retrieval, cosine math, empty results, cache invalidation, grounded
API responses, general-mode bypass, invalid input, missing settings, provider
errors, optional tracing, and Streamlit chat/history/connection errors. They do
not measure actual Azure model quality or prove delivery to a Langfuse server.

Academy concepts demonstrated: prompt engineering and evidence grounding; RAG
chunking and metadata; embeddings/cosine versus lexical search; deterministic
routing and hybrid scoring; bounded conversational memory; API/UI separation;
environment-based secret configuration; observability; and offline testing through
injected model boundaries. The implementation intentionally avoids LangChain or
LangGraph because a few functions express this pipeline clearly.

## Limits and next improvements

This is a small local demo with a curated corpus, not a comprehensive
maritime reference. The identifier parser covers common SOLAS/ISM/Code forms, not
every legal citation syntax. Retrieval can miss context or rank irrelevant notes;
models can hallucinate despite prompts and label checks. A source label is not a
proof of entailment. Do not use this app to direct emergencies or decide compliance.
The deployment must support the chosen chat parameters; API availability, model
quality and embedding thresholds need a real-credential smoke test in your setup.
There is no automatic cross-turn retrieval rewrite or persistent chat, and refresh
or server restart can lose session history. Keep `data/` small and trusted.

Next steps: expert-reviewed licensed notes with version/applicability metadata,
a labeled maritime retrieval evaluation set, threshold tuning, better identifier
parsing and follow-up query rewriting, and claim/citation evaluation. Add complexity
only after the demo has been evaluated.

For a standalone personal GitHub project, copy this directory's source files into
a repository named `maritime-safety-copilot`. Exclude `.env`, `.venv`, `.cache`,
logs and all generated files using the supplied `.gitignore`. No git commit,
remote repository creation or publication is performed by the app.

## Implementation references

- [Official OpenAI documentation: embeddings and cosine search](https://developers.openai.com/api/docs/guides/embeddings).
- [Microsoft: OpenAI and Azure OpenAI endpoints](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/how-to/switching-endpoints?view=foundry-classic).
- [Langfuse Python instrumentation](https://langfuse.com/docs/observability/sdk/instrumentation).
- [PyYAML: safe YAML parsing](https://pyyaml.org/wiki/PyYAMLDocumentation).
- [IMO introduction to the ISM Code](https://www.imo.org/en/OurWork/HumanElement/Pages/ISMCode.aspx).
