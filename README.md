# 🩺 Health Information Assistant

A Streamlit app that uses Google's **Gemini Flash** model to give **general health information**: possible explanations, precautions, and guidance on when to see a healthcare professional.

> ⚠️ **Disclaimer:** This app is for general education only. It does **not** provide a diagnosis, treatment plan or medical advice, and it does **not** replace a doctor, pharmacist or other qualified professional. In an emergency, call your local emergency number (Pakistan: 1122 / 115).

## Features

- Free-text question or symptom description, plus optional age group and duration
- Structured answer: quick summary, general information, possible explanations (clearly *not* a diagnosis), self-care and precautions, and when to see a doctor
- **Local safety check** for emergency wording (e.g. chest pain, trouble breathing) and self-harm wording. It shows an urgent banner even if the AI call fails
- Safety rules in the system prompt: no definitive diagnosis, no drug doses, always list red-flag symptoms, refuse non-health questions, and ignore attempts to override the rules
- Replies in the user's language (English, Urdu, etc.)
- Input limits, a short cooldown between requests, and friendly error messages
- API key kept in secrets, never in code

## Project structure

```
health-info-assistant/
├── app.py            # Streamlit app
├── requirements.txt  # Python dependencies
└── README.md
```

## Run locally

**1. Get a free Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey).

**2. Create a virtual environment and install dependencies**

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

**3. Add your key.** Create the file `.streamlit/secrets.toml` in the project folder:

```toml
GEMINI_API_KEY = "paste-your-key-here"
# Optional: choose another Flash model
# GEMINI_MODEL = "gemini-2.5-flash"
```

(Alternatively set an environment variable named `GEMINI_API_KEY`.)

**4. Start the app**

```bash
streamlit run app.py
```

It opens at http://localhost:8501.

## Push to GitHub

Create a `.gitignore` first so your key is never uploaded:

```bash
printf "venv/\n__pycache__/\n.env\n.streamlit/secrets.toml\n" > .gitignore
```

Then, inside the project folder:

```bash
git init
git add .
git status            # check that secrets.toml is NOT listed
git commit -m "Initial commit: health information assistant"
git branch -M main
```

On github.com click **New repository**, name it (e.g. `health-info-assistant`), leave it **empty** (no README/.gitignore), and create it. Then:

```bash
git remote add origin https://github.com/<your-username>/health-info-assistant.git
git push -u origin main
```

GitHub no longer accepts account passwords over HTTPS. When asked to sign in, use a **Personal Access Token** (GitHub → Settings → Developer settings → Tokens), or sign in through GitHub Desktop or `gh auth login`.

## Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub (authorize access to your repository).
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Choose your repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "paste-your-key-here"
   ```
5. Click **Deploy**. After a minute or two you get a public `*.streamlit.app` link.

To update the app later, just `git add . && git commit -m "..." && git push`. Streamlit redeploys automatically. Secrets can be changed any time under the app's **Settings → Secrets**.

## Notes and limits

- The free Gemini tier has rate limits. If you see a "busy or quota" message, wait a minute.
- Don't type personal identifiers (name, ID number, phone) into the app.
- The keyword safety check is a simple safety net, not a medical triage tool. It can miss some wording and can occasionally trigger on harmless text.
- AI answers can be wrong. Treat them as a starting point for questions to ask a professional.

## Troubleshooting

- **`ImportError` / `ModuleNotFoundError` on Streamlit Cloud:** make sure `requirements.txt` is in the **root** of the repository (next to `app.py`, not inside a sub-folder), then reboot the app from **Manage app → Reboot**. This app talks to Gemini through plain HTTPS (`requests`), so it needs no Google SDK.
- **"No Gemini API key found":** add `GEMINI_API_KEY` under **Settings → Secrets** (cloud) or in `.streamlit/secrets.toml` (local).
- **"API key looks invalid":** create a new key in Google AI Studio and paste it again.
