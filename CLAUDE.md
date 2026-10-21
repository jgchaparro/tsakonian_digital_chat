# Tsakonian Digital Chat

This is a standalone repository. The Streamlit interface retains the Tsakonian
Digital Lab branding and shares the dictionary's visual style.

## Architecture

- `app.py`: chat UI, assistant settings, saved pipeline stages and retries.
- `pipeline.py`: one local Leonida translator for both directions; Greek replies
  use local Google Gemma 4 E4B by default. KriKri and NVIDIA NIM are optional.
- `ui.py`, `assets/`, `.streamlit/config.toml`: presentation and original brand assets.
- `test_lab.py`: mocked pipeline and Streamlit integration checks.

## Working conventions

- Preserve Leonida's exact system/user translation prompts and raw Llama 3.1
  template. Both translation directions must use the same configured model.
- Preserve completed stages when retrying; never append partial turns to history.
- Keep Greek text directly beneath each speaker's Tsakonian text with matching
  spacing. Completed turns have no pipeline panel.
- Keep `.env`, secrets, virtual environments, logs and model weights out of Git.
- Commit and push only when requested.

## Local commands

Run from this repository's directory in PowerShell:

```powershell
.venv/Scripts/python.exe -X utf8 -m unittest -q test_lab
.venv/Scripts/python.exe -m streamlit run app.py
```

See `README.md` and `.env.example` for setup and model configuration. Network
calls are mocked in the test suite; automated checks need no API credentials or
running models.
