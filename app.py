import logging
import os
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import tiktoken
from dotenv import dotenv_values
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Change this one line to "text-embedding-3-small" for the smaller, cheaper model.
MODEL_NAME = "text-embedding-3-small"
# Specs OpenAI publishes for each model. The architecture, parameter count and weights are not disclosed.
OPENAI_MODELS = {
    "text-embedding-3-small": {"dimensions": 1536, "usd_per_1m_tokens": 0.02},
    "text-embedding-3-large": {"dimensions": 3072, "usd_per_1m_tokens": 0.13},
}
MAX_INPUT_TOKENS = 8192
USD_PER_1M_TOKENS = OPENAI_MODELS[MODEL_NAME]["usd_per_1m_tokens"]
MODEL_URL = f"https://platform.openai.com/docs/models/{MODEL_NAME}"
MODEL_LICENSE = "Proprietary (OpenAI API terms)"
COST_HELP = (
    f"OpenAI charges ${USD_PER_1M_TOKENS} per 1 million input tokens for {MODEL_NAME}. "
    "Cost = tokens sent × price per token; the returned vector itself is free."
)
ENV_FILE = Path(__file__).resolve().parent / ".env"
LOG_FILE = Path(__file__).resolve().parent / "app.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | session=%(session_id)s | %(name)s:%(lineno)d | %(message)s"


class _SessionIdDefault(logging.Filter):
    # Library log records have no session id; give them a placeholder so the format string works.
    def filter(self, record):
        if not hasattr(record, "session_id"):
            record.session_id = "-"
        return True


def setup_logging():
    # Streamlit re-executes this script on every interaction, so only attach handlers once per process.
    logger = logging.getLogger("embedding_app")
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_SessionIdDefault())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_SessionIdDefault())

    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    # Record the OpenAI client's retry messages and its HTTP request lines (URL + status, no API key) in app.log too.
    for name in ("openai", "httpx2"):
        lib_logger = logging.getLogger(name)
        lib_logger.setLevel(logging.INFO)
        lib_logger.addHandler(file_handler)

    logger.info("Logging initialised; writing to %s", LOG_FILE)
    return logger


logger = setup_logging()

st.set_page_config(page_title="OpenAI Text → Numbers: Embeddings", page_icon="🧠", layout="wide")

is_new_session = "session_id" not in st.session_state
if is_new_session:
    st.session_state["session_id"] = uuid.uuid4().hex[:8]
log = logging.LoggerAdapter(logger, {"session_id": st.session_state["session_id"]})
if is_new_session:
    log.info("New browser session started")
log.debug("Script run started | converted=%s", st.session_state.get("converted", False))


def get_api_key():
    # .streamlit/secrets.toml wins, then the .env file next to app.py, then the OPENAI_API_KEY environment variable.
    # The .env file is re-read on each call, so a newly pasted key works without restarting the app.
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return dotenv_values(ENV_FILE).get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")


def token_text(piece):
    # tiktoken works on bytes, so one token can be part of a multi-byte character (common in Hindi).
    # Show those pieces as hex bytes, and show spaces as ␣ because they are part of the token.
    try:
        return piece.decode("utf-8").replace(" ", "␣")
    except UnicodeDecodeError:
        return "bytes " + piece.hex(" ")


def format_cost(usd):
    # One sentence costs a tiny fraction of a cent, so keep enough decimals to show it.
    return f"${usd:.8f}".rstrip("0").rstrip(".") if usd else "$0.00"


@st.cache_resource
def load_model():
    # The model runs on OpenAI's servers, so "loading" means creating the API client and the local tokenizer.
    # Returns both plus the model's published specs for the model card. Fails fast if no API key is set.
    logger.info("Setting up OpenAI client and tokenizer for '%s'", MODEL_NAME)
    start = time.perf_counter()
    client = OpenAI(api_key=get_api_key(), timeout=30)
    # tiktoken downloads the tokenizer's vocabulary file on first use and caches it afterwards.
    tokenizer = tiktoken.encoding_for_model(MODEL_NAME)
    load_seconds = time.perf_counter() - start

    specs = {
        "dimensions": OPENAI_MODELS[MODEL_NAME]["dimensions"],
        "max_seq_length": MAX_INPUT_TOKENS,
        "usd_per_1m_tokens": USD_PER_1M_TOKENS,
        "tokenizer": tokenizer.name,
        "vocab_size": tokenizer.n_vocab,
        "load_seconds": load_seconds,
    }
    logger.info("OpenAI client for '%s' ready in %.2fs | specs=%s", MODEL_NAME, load_seconds, specs)
    return client, tokenizer, specs


st.title("🧠 Text → Numbers: See Embeddings in Action")
st.markdown(f":material/memory: **Embedding model in use:** [`{MODEL_NAME}`]({MODEL_URL}) (OpenAI API)")
st.write(
    "Type text and explore how an embedding model turns it into a list of numbers. "
    "Each conversion calls the OpenAI API, so you need an internet connection and an API key."
)

with st.expander("🔎 The journey in one glance", expanded=True):
    st.markdown(
        "**Your sentence** → **tokens** → **token IDs** → "
        "**learned vector for each token (model internals)** → **sentence embedding**"
    )
    st.caption(
        "Note: this app shows real token IDs from the model's tokenizer and the real final embedding. "
        "The model's internal token vectors are not the same as token IDs."
    )

default_text = "I love mangoes"
text = st.text_input("✍️ Enter a sentence", value=default_text, max_chars=500)

with st.sidebar:
    st.header(":material/memory: Model in use")
    st.markdown(f"[`{MODEL_NAME}`]({MODEL_URL})")
    try:
        with st.spinner("Connecting to OpenAI..."):
            _, _, specs = load_model()
    except Exception:
        log.exception("Failed to set up the OpenAI client for '%s'", MODEL_NAME)
        st.error(
            "Could not set up the OpenAI client. Check that `OPENAI_API_KEY` is set. "
            f"Details were written to `{LOG_FILE.name}`."
        )
    else:
        m1, m2 = st.columns(2)
        m1.metric("Dimensions", specs["dimensions"], border=True, help="Length of the vector produced for each sentence.")
        m2.metric("Max input", f"{specs['max_seq_length']} tokens", border=True, help="Longer inputs are rejected by the API.")
        m3, m4 = st.columns(2)
        m3.metric("Parameters", "N/A", border=True, help="OpenAI does not disclose the model's parameter count.")
        m4.metric("Weights size", "Cloud", border=True, help="The weights stay on OpenAI's servers; nothing is downloaded to this machine.")
        m5, m6 = st.columns(2)
        m5.metric("Cost / 1M tokens", f"${specs['usd_per_1m_tokens']}", border=True, help=COST_HELP)
        m6.metric("Setup time", f"{specs['load_seconds']:.1f} s", border=True, help="Time this server took to create the API client and load the tokenizer.")

        details = pd.DataFrame(
            [
                ("Architecture", "Not disclosed by OpenAI"),
                ("Tokenizer", f"{specs['tokenizer']} (byte-level BPE)"),
                ("Vocabulary", f"{specs['vocab_size']:,} tokens"),
                ("Adjustable dimensions", "Yes (API dimensions option)"),
                ("Normalized output", "Yes"),
                ("Similarity function", "cosine"),
                ("Price", f"${specs['usd_per_1m_tokens']} / 1M tokens"),
                ("Running on", "OpenAI API (cloud)"),
                ("License", MODEL_LICENSE),
                ("Language", "Multilingual"),
            ],
            columns=["Property", "Value"],
        )
        st.dataframe(details, hide_index=True, alt=f"Technical details of the {MODEL_NAME} model")

if st.button("✨ Convert text to numbers", type="primary"):
    log.info("Convert clicked | text_length=%d", len(text))
    log.debug("Input text: %r", text)
    if not text.strip():
        log.warning("Convert aborted: input text is empty")
        st.warning("Please enter some text first.")
        st.stop()

    try:
        with st.spinner("Calling the OpenAI API to create your embedding..."):
            start = time.perf_counter()
            client, tokenizer, specs = load_model()

            # Token IDs are real IDs from the same tiktoken tokenizer the OpenAI model uses.
            # OpenAI adds no special tokens, and the 500-character input cap stays far below the token limit,
            # so the table shows exactly the tokens the model sees.
            token_ids = tokenizer.encode(text)
            tokens = [token_text(tokenizer.decode_single_token_bytes(token_id)) for token_id in token_ids]
            response = client.embeddings.create(model=MODEL_NAME, input=text)
            vector = np.asarray(response.data[0].embedding, dtype=float)
            elapsed = time.perf_counter() - start
    except Exception:
        log.exception("Failed to create embedding for input text")
        st.error(
            "Something went wrong while calling the OpenAI API. Check your API key and internet connection. "
            f"Details were written to `{LOG_FILE.name}`."
        )
        st.stop()

    billed_tokens = response.usage.total_tokens
    cost = billed_tokens * specs["usd_per_1m_tokens"] / 1e6
    if billed_tokens != len(token_ids):
        log.warning("Token count mismatch | tiktoken=%d | api=%d", len(token_ids), billed_tokens)
    log.info(
        "Embedding created in %.3fs | tokens=%d | dimensions=%d | l2_norm=%.4f | cost_usd=%.8f",
        elapsed, billed_tokens, len(vector), float(np.linalg.norm(vector)), cost,
    )
    log.debug("Tokens: %s", tokens)
    log.debug("Token IDs: %s", token_ids)
    log.debug("Embedding (first 10 values): %s", np.round(vector[:10], 6).tolist())

    st.session_state["tokens"] = tokens
    st.session_state["token_ids"] = token_ids
    st.session_state["vector"] = vector
    st.session_state["embed_seconds"] = elapsed
    st.session_state["embed_cost"] = cost
    st.session_state["max_tokens"] = specs["max_seq_length"]
    st.session_state["converted_text"] = text
    st.session_state["converted"] = True

if st.session_state.get("converted", False):
    tokens = st.session_state["tokens"]
    token_ids = st.session_state["token_ids"]
    vector = st.session_state["vector"]
    st.success(f"Done! Embedding created with `{MODEL_NAME}` for: “{st.session_state['converted_text']}”")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Embedding dimensions", len(vector), border=True)
    s2.metric(
        "Tokens used", f"{len(tokens)} / {st.session_state['max_tokens']}", border=True,
        help="Tokens in your sentence out of the model's maximum. Unlike BERT-style models, "
        "OpenAI's tokenizer adds no special [CLS] or [SEP] markers.",
    )
    s3.metric(
        "Time taken", f"{st.session_state['embed_seconds'] * 1000:.0f} ms", border=True,
        help="Tokenization plus the round trip to the OpenAI API for this sentence. "
        "The first run also includes setting up the client and tokenizer.",
    )
    s4.metric("Cost", format_cost(st.session_state["embed_cost"]), border=True, help=COST_HELP)

    st.subheader("1️⃣ Tokenization + Token IDs")
    st.write("The tokenizer splits the text into pieces. Each piece gets a numeric ID.")
    token_df = pd.DataFrame({
        "Token (text piece)": tokens,
        "Token ID": token_ids
    })
    st.dataframe(token_df, width="stretch", hide_index=True)
    st.caption(
        "Token pieces may be whole words, parts of words, punctuation, or raw bytes of a character. "
        "␣ marks a space that is part of the token."
    )

    st.subheader("2️⃣ The final sentence embedding")
    st.write(
        f"The model `{MODEL_NAME}` represents your entire sentence using **{len(vector)} numbers**. "
        "These numbers are learned by the model during training."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.code("[" + ", ".join(f"{x:.4f}" for x in vector[:12]) + (", ...]" if len(vector) > 12 else "]"))
        st.caption("Showing the first 12 values above; the full vector is available below.")
    with col2:
        preview_count = min(40, len(vector))
        chart_df = pd.DataFrame({
            "Dimension": [f"n{i}" for i in range(preview_count)],
            "Value": vector[:preview_count]
        }).set_index("Dimension")
        st.bar_chart(chart_df)
        st.caption(f"Visualization of the first {preview_count} dimensions. Each dimension is one position in the vector.")

    with st.expander("📋 Show the complete embedding vector"):
        st.code("[" + ", ".join(f"{x:.6f}" for x in vector) + "]")
        if st.download_button(
            "⬇️ Download embedding as CSV",
            data=pd.DataFrame({"dimension": range(len(vector)), "value": vector}).to_csv(index=False),
            file_name="sentence_embedding.csv",
            mime="text/csv"
        ):
            log.info("Embedding CSV downloaded | dimensions=%d", len(vector))

    st.subheader("3️⃣ Compare meanings")
    st.write("Try two sentences. Cosine similarity gives a rough measure of how close their embeddings are.")
    c1, c2 = st.columns(2)
    with c1:
        sentence_a = st.text_input("Sentence A", value=text, key="sentence_a")
    with c2:
        sentence_b = st.text_input("Sentence B", value="I really enjoy eating mangoes", key="sentence_b")

    if st.button("🔗 Compare these sentences"):
        log.info("Compare clicked | len_a=%d | len_b=%d", len(sentence_a), len(sentence_b))
        log.debug("Sentence A: %r | Sentence B: %r", sentence_a, sentence_b)
        if not sentence_a.strip() or not sentence_b.strip():
            log.warning("Compare aborted: one or both sentences are empty")
            st.warning("Enter both sentences to compare.")
        else:
            try:
                start = time.perf_counter()
                # The client is cached, so this is instant after the first setup. It must be fetched here
                # because this button click is a fresh script run where the Convert block did not execute.
                client, _, specs = load_model()
                # Both sentences go in one API request. OpenAI returns vectors already normalized to length 1.
                response = client.embeddings.create(model=MODEL_NAME, input=[sentence_a, sentence_b])
                emb_a, emb_b = (item.embedding for item in response.data)
                score = float(cosine_similarity([emb_a], [emb_b])[0][0])
                elapsed = time.perf_counter() - start
                billed_tokens = response.usage.total_tokens
                cost = billed_tokens * specs["usd_per_1m_tokens"] / 1e6
            except Exception:
                log.exception("Failed to compare sentences")
                st.error(f"Something went wrong while comparing the sentences. Details were written to `{LOG_FILE.name}`.")
            else:
                log.info(
                    "Cosine similarity computed in %.3fs | score=%.4f | tokens=%d | cost_usd=%.8f",
                    elapsed, score, billed_tokens, cost,
                )
                st.metric("Cosine similarity", f"{score:.3f}")
                st.progress(max(0.0, min(1.0, (score + 1) / 2)))
                st.caption(
                    "Closer to 1 usually means more semantically similar; closer to 0 means less similar. "
                    "This is a useful intuition, not a perfect measure of meaning."
                )
                st.caption(f"Computed in {elapsed * 1000:.0f} ms · {billed_tokens} tokens · cost {format_cost(cost)} (OpenAI API)")

    with st.expander("🧩 What just happened?"):
        st.markdown(
            """
            1. **Tokenizer:** splits the sentence into pieces.
            2. **Token IDs:** gives each piece an ID so the model can process it.
            3. **Embedding model:** uses learned numerical patterns to represent meaning and context.
            4. **Sentence embedding:** produces one fixed-size vector for the whole sentence.

            **Remember:** the token IDs are not the embedding. The final vector is the embedding you commonly store in a vector database for semantic search or RAG.
            """
        )
else:
    st.info("Enter a sentence above and click **Convert text to numbers** to explore the real output.")
    st.markdown("### Example sentences to try")
    st.write("- `I love mangoes`")
    st.write("- `I enjoy eating fruit`")
    st.write("- `The train arrives at noon`")
    st.caption("Similar ideas may have closer vectors, but the exact values depend on the model.")
