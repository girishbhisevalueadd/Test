# Text → Numbers: Embeddings (Streamlit Demo)

A beginner-friendly app that shows real tokenizer pieces, token IDs, and a real sentence embedding from OpenAI's `text-embedding-3-large`.

## Requirements
- Python 3.10–3.12 recommended
- An OpenAI API key with billing enabled
- Internet connection (every conversion calls the OpenAI API)

## Run on Windows

1. Extract the ZIP file.
2. Open the `embedding_streamlit_demo` folder.
3. Open Command Prompt or PowerShell in this folder.
4. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

5. Install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Paste your OpenAI API key into the `.env` file in this folder (already git-ignored):

```
OPENAI_API_KEY=sk-...
```

Alternatively, put `OPENAI_API_KEY = "sk-..."` in `.streamlit/secrets.toml`, or set it as an environment variable in the same terminal before starting the app:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

7. Start the app:

```bash
streamlit run app.py
```

8. Streamlit should open a browser tab, usually at `http://localhost:8501`.

## What to try
- Enter `I love mangoes`, then click **Convert text to numbers**.
- Inspect the token table and embedding vector.
- Expand **Show the complete embedding vector**.
- Compare `I love mangoes` with `I really enjoy eating mangoes`, then compare it with `The train arrives at noon`.
- Try a cross-language pair such as `I love mangoes` and `मुझे आम अच्छे लगते हैं`.

## Important beginner notes
- Token IDs are labels/identifiers; they do not represent the meaning of a word.
- The final sentence embedding is a list of 3,072 real numbers for this model (1,536 for `text-embedding-3-small`; change `MODEL_NAME` in `app.py` to switch).
- The bar chart displays the first 40 dimensions only.
- Each conversion is billed by OpenAI per input token ($0.13 per 1M tokens); the app shows the cost of every request.
