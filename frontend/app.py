"""Streamlit client. All model, search and tracing work belongs to the API."""

import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

from frontend.streaming import StreamError, answer_tokens
from frontend.uploads import save_markdown

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Maritime Safety Copilot", page_icon="⚓", layout="centered")
hero_image = Path(__file__).resolve().parents[1] / "assets" / "twilight_cargo_ship_at_sea.png"
st.image(str(hero_image), use_container_width=True)
st.title("⚓ Maritime Safety Copilot")
st.caption("AI-powered maritime safety knowledge assistant")
st.info(
    "Educational demo · Original study notes, not authoritative regulatory text. "
    "Use current official material and approved vessel procedures for operational decisions."
)

with st.sidebar:
    st.header("Your workspace")
    mode_label = st.radio("Answer mode", ["Knowledge Base", "General LLM"])
    mode = "knowledge_base" if mode_label == "Knowledge Base" else "general"
    retrieval_on = st.toggle("Retrieval", value=True, disabled=mode == "general")
    top_k = st.slider("Maximum sources", 1, 6, 3, disabled=mode == "general" or not retrieval_on)
    if mode == "general":
        st.caption("General LLM answers bypass the knowledge base and have no retrieved sources.")
    elif not retrieval_on:
        st.caption("Without retrieval, Knowledge Base mode cannot supply a grounded answer.")
    if st.button("Clear chat", width="stretch"):
        st.session_state["chats"] = {"knowledge_base": [], "general": []}
        st.rerun()
    st.divider()
    with st.expander("Add a knowledge file (optional)"):
        uploaded = st.file_uploader("Markdown knowledge file", type=["md"], max_upload_size=1)
        st.caption("UTF-8 Markdown, up to 1 MB. Use curated notes you have permission to add.")
        if st.button("Save to knowledge base", disabled=uploaded is None, key="save_knowledge"):
            try:
                saved = save_markdown(uploaded.name, uploaded.getvalue())
            except ValueError as exc:
                st.error(str(exc))
            except OSError:
                st.error(
                    "Could not save the file. Check write access to the local data/ directory."
                )
            else:
                st.success(
                    f"Saved as {saved.name}. The backend reads data/ on the next retrieval. "
                    "Embeddings refresh on the next semantic/hybrid retrieval; restarting "
                    "the backend also clears its cache. No restart is required for this upload."
                )

if "chats" not in st.session_state:
    st.session_state["chats"] = {"knowledge_base": [], "general": []}
chat = st.session_state["chats"][mode]


def show_response(result: dict):
    st.markdown(result["answer"])
    for warning in result.get("warnings", []):
        st.caption(warning)
    sources = result["sources"]
    if sources:
        st.caption("Retrieved demo sources · Labels apply to this answer only")
        for source in sources:
            with st.expander(f"[{source['citation']}] {source['section']}"):
                if source.get("title"):
                    st.caption(source["title"])
                st.caption(f"{source['source']} · {source['chunk_id']}")
                st.text(source["excerpt"])
                if source.get("metadata"):
                    st.json(source["metadata"], expanded=False)
    with st.expander("Technical details"):
        details = result["technical_details"]
        st.write(
            f"Method: **{details['retrieval_method']}** · "
            f"Chunks: **{details['chunks_retrieved']}** · "
            f"Latency: **{details['latency_ms']:,.0f} ms**"
        )
        if sources:
            st.dataframe(
                [
                    {
                        "Source": s["citation"],
                        "Rank score": s["score"],
                        "Cosine similarity": s["similarity"],
                        "Keyword score": s["keyword_score"],
                    }
                    for s in sources
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Exact: keyword score. Semantic: cosine similarity. "
                "Hybrid: 60% nonnegative cosine + 40% keyword score. "
                "Scores measure retrieval relevance, not answer confidence."
            )


for message in chat:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_response(message["result"])
        else:
            st.markdown(message["content"])

st.caption("Try a prompt")
suggested_question = None
examples = [
    ("SOLAS IX", "What is SOLAS IX?"),
    ("Emergency preparedness", "How does ISM Code section 8 support emergency preparedness?"),
    ("Permit to work", "What is the difference between a permit to work and a checklist?"),
]
for column, (label, prompt) in zip(st.columns(len(examples)), examples):
    if column.button(label, help=prompt, key=f"try_{label}", width="stretch"):
        suggested_question = prompt

question = st.chat_input("Ask about maritime safety…", max_chars=2000) or suggested_question
if question and question.strip():
    with st.chat_message("user"):
        st.markdown(question)
    history = [{"role": item["role"], "content": item["content"]} for item in chat[-8:]]
    with st.chat_message("assistant"):
        provisional = st.empty()
        try:
            result = {}
            with provisional.container():
                st.caption("Generating… The final answer and sources appear when complete.")
                st.write_stream(
                    answer_tokens(
                        f"{BACKEND_URL}/ask/stream",
                        {
                            "question": question,
                            "mode": mode,
                            "retrieval_enabled": retrieval_on and mode == "knowledge_base",
                            "top_k": top_k,
                            "history": history,
                        },
                        result,
                    )
                )
            provisional.empty()
            show_response(result)
        except StreamError as exc:
            provisional.empty()
            st.error(str(exc))
        except httpx.HTTPStatusError as exc:
            provisional.empty()
            try:
                detail = exc.response.json().get("detail", "Backend request failed.")
            except ValueError:
                detail = "Backend request failed. Check the backend terminal."
            st.error(f"Request failed ({exc.response.status_code}): {detail}")
        except httpx.RequestError:
            provisional.empty()
            st.error(
                "Cannot reach the backend or the request timed out. Start FastAPI on "
                "127.0.0.1:8000 and check BACKEND_URL in .env, then retry."
            )
        except (ValueError, KeyError, TypeError):
            provisional.empty()
            st.error(
                "The backend returned an unexpected response. Check that both apps use this project."
            )
        else:
            chat.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": result["answer"], "result": result},
                ]
            )
            # Six recent exchanges on screen; at most four sent to the backend.
            del chat[:-12]
