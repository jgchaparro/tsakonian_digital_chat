# Tsakonian Digital Chat

## Quickstart

Needs Python 3 and a running [Ollama](https://ollama.com/). The translator is
[Leonida](https://huggingface.co/jgchaparro/Llama-Leonida-GGUF) (see
[Run on this computer](#run-on-this-computer)). In PowerShell:

```powershell
git clone https://github.com/jgchaparro/tsakonian_digital_chat.git
cd tsakonian_digital_chat
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
copy .env.example .env
ollama pull hf.co/jgchaparro/Llama-Leonida-GGUF:Q5_K_M
ollama create leonida -f Modelfile
ollama pull gemma4:e4b
.venv/Scripts/python.exe -m streamlit run app.py
```

The chat opens at <http://localhost:8501>. After the first setup, only the last
line is needed. To use NVIDIA NIM instead of Ollama, put your key in
`NVIDIA_API_KEY` in `.env` and pick **NVIDIA NIM** in the sidebar.

## About

Repository: [jgchaparro/tsakonian_digital_chat](https://github.com/jgchaparro/tsakonian_digital_chat).
The app uses the **Tsakonian Digital Lab** branding in its interface.

A small Streamlit chat: Tsakonian → Greek with local Leonida → Greek reply
from local Google Gemma 4 E4B → Tsakonian with the same local Leonida model.
KriKri and NVIDIA NIM remain selectable alternatives. Greek text appears beneath
each message; pipeline details disappear when a turn completes. NIM receives
the last ten completed Greek conversation turns. **Clear conversation** resets it.
The Greek translation appears directly below each user's message as soon as the
first translation finishes, in smaller grey text. It stays visible during failures
and in the completed conversation. The Greek source of each Leonida reply also
appears directly beneath it, with the same smaller grey text and compact spacing.

## Run on this computer

The local project folder is `Tsakonio/tsakonian_digital_chat`. For a new checkout,
clone the private repository with your GitHub account:

```powershell
git clone https://github.com/jgchaparro/tsakonian_digital_chat.git
cd tsakonian_digital_chat
python -m venv .venv
```

From the project directory in PowerShell:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
ollama pull hf.co/jgchaparro/Llama-Leonida-GGUF:Q5_K_M
ollama create leonida -f Modelfile
ollama pull gemma4:e4b
.venv/Scripts/python.exe -m streamlit run app.py
```

Ollama must be running. The Modelfile builds `leonida` from the Leonida
Q5_K_M GGUF and sets the Llama 3.1 chat template. Both translation directions
call only `leonida`. To use the earlier Leonida Beta instead, change the Modelfile's
`FROM` line to `hf.co/jgchaparro/Llama-Leonida-Beta-GGUF:Q5_K_M` and run
`ollama create leonida -f Modelfile` again.

The app reads `NVIDIA_API_KEY` from the environment or this directory's `.env`.
The key is needed only when NVIDIA NIM is selected; local assistants work without it.
To configure a local file, copy `.env.example` to `.env` and fill in the key.
Existing environment variables take precedence except for the provider and model
defaults: local `.env` choices are reread on every Streamlit rerun. Edited provider
or local-model defaults update open sessions once, retaining subsequent selections
made in the sidebar and all saved conversation results.
This lets **Retry failed step** use a replacement
model without losing the completed translation. Never commit `.env`.

Optional settings are listed in `.env.example`: `NIM_MODEL`, `NIM_MODELS`
(extra comma-separated NIM models for the sidebar selector), `LEONIDA_MODEL`,
`OLLAMA_BASE_URL` (native URL or the previous `/v1` URL), and `OLLAMA_KEEP_ALIVE`.
`NIM_MAX_TOKENS` defaults to 2048 and `NIM_REASONING_EFFORT` to `low`.
The reasoning option is sent only to GPT-OSS models. The token allowance includes
their internal reasoning, so it should not be reduced just to shorten the visible
answer. The Greek system prompt separately requests 1–3 sentences, up to 60 words.
The optional NIM route uses `nvidia/nemotron-3-ultra-550b-a55b`. It responded to the
Leonidio question and a Greek follow-up in live checks on 2026-10-21.
`openai/gpt-oss-20b` stopped responding: direct and streaming requests timed out
after 100 seconds. An invocation with `NVCF-POLL-SECONDS: 15` returned HTTP 504
and `nvcf-status: errored` after 17.7 seconds. NVIDIA's documentation describes
this status as a worker not picking up the request within the polling window.
Other model endpoints worked with the same API key, so the failed model's
inference availability was the immediate blocker. Several older models returned
account-specific 404 responses even though NIM still listed them.
The default keep-alive is one hour after each translation. The first translation
after startup or unloading still needs to load the weights; subsequent directions
reuse that same loaded model. Local translations use a 180-second read timeout;
NIM requests use a 60-second timeout with SDK retries disabled. A token-limited
NIM response is regenerated once with twice the budget, capped at 4096 tokens.
Network failures remain available for manual retry, preserving completed steps.

## Translation contract

Every translation is an independent Ollama `/api/generate` request with exactly
these message contents, serialized using the model's Llama 3.1 chat template:

```text
System: You are a precise translation assistant. Output only the translation, nothing else.
User: Translate from {origin} to {target}:\n\n{source}
```

`origin` and `target` are `Greek` and `Tsakonian`, exchanged for the return
translation. The `\n\n` represents two actual newline characters. Translation
requests use temperature zero and carry no chat history. `pipeline.py` includes
the template's beginning-of-text token, role headers, end-of-turn tokens, assistant
generation header, and the tokenizer's fixed date prelude used during training.

Using `raw: true` bypasses the template embedded in the GGUF and sends the exact
serialized prompt the model was trained on, so a single loaded model translates
both ways. (Leonida Beta's direction-specific GGUFs embedded templates that
prepended a fixed direction; the Leonida GGUF has the standard Llama 3.1
template.)

Empty responses or responses that remain token-limited stop the pipeline; an
unfinished answer is never translated or added to history. Completed steps are
saved in the current browser session. If a step fails, **Retry failed step**
resumes there and runs any remaining steps, reusing earlier translations and
assistant replies. Completed steps remain visible while waiting to retry. New messages
are disabled until the pending message finishes or you choose **Discard message**.
**Clear conversation** clears both completed turns and the pending message.
Pending results are held in memory and do not survive a new browser session or
server restart. Failed turns are added to conversation history only after all
three steps succeed. Pipeline completion verifies connectivity and
request handling; translation quality still needs review by a Tsakonian speaker.

## Greek reply assistant

Settings live in the sidebar. **Reply assistant** switches between **Ollama**
(the default) and **NVIDIA NIM**; **Model** changes with the provider. For Ollama it
lists Google Gemma 4 E4B (default), ILSP KriKri 8B (Greek), the configured model and
every other installed model reported by `/api/tags` (refreshed every 30 seconds;
Leonida itself is excluded). For NIM it lists `NIM_MODEL` plus `NIM_MODELS`.
Without `NVIDIA_API_KEY`, choosing NIM shows a warning and disables sending
instead of stopping the app. **Show Greek translation** hides the Greek lines for
practice, and **Clear conversation** is also in the sidebar.
You can also switch providers or local models after a failed Greek
reply and press **Retry failed step**; the saved initial translation is reused.
Each completed turn records which assistant produced its Greek reply.

This computer has both local assistants installed. On a new installation,
download the default Google model; download KriKri only if desired:

```powershell
ollama pull gemma4:e4b
ollama pull hf.co/ilsp/Llama-Krikri-8B-Instruct-GGUF:Q4_K_M
```

Local mode uses the selected assistant for the Greek conversation and the single Leonida model
for both translation directions. It sends no Greek reply requests to NVIDIA.
Both assistants get the same Greek system prompt and the last ten completed Greek turns,
through Ollama's native `/api/chat`; Leonida keeps its exact translation prompts.
Gemma 4 runs with thinking disabled to keep short conversation replies responsive.

The official KriKri Q4_K_M weights are about 5 GB; Gemma 4 E4B needs additional
disk space and memory for its model and multimodal components. With
16 GB total RAM and other applications open, both models may not fit in physical
memory together. Ollama may evict and reload models, or Windows may page memory;
local mode can consequently be slower than NVIDIA mode. Freeing memory helps
keep the selected assistant and Leonida loaded together. NVIDIA mode needs only Leonida locally.

Default `.env` settings are `GREEK_BACKEND=ollama`, `OLLAMA_CHAT_MODEL=gemma4:e4b`,
and `OLLAMA_CHAT_KEEP_ALIVE=5m`. Choose `GREEK_BACKEND=nim` for the optional cloud
route. The old `KRIKRI_KEEP_ALIVE` setting is still accepted as a fallback.
Local assistants are retained for five minutes after their last reply;
Leonida's keep-alive remains independently configurable.

Cloud checks on 2026-10-21 found that Google Gemma 3 4B/12B returned account-specific
404 responses, while Gemma 4 31B timed out at 40 and 50 seconds; a bounded polling
request returned HTTP 504. Listing a model does not establish working inference
access. Local Gemma avoids those endpoints and requires no additional API key.

Source: [ILSP's official KriKri GGUF model](https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct-GGUF).
Google references: [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
and [Ollama Gemma 4 E4B](https://ollama.com/library/gemma4:e4b).

Ollama references: [generation API and raw prompts](https://docs.ollama.com/api/generate),
[Modelfile templates](https://docs.ollama.com/modelfile).

NVIDIA reference: [GPT-OSS generation limits and reasoning effort](https://docs.api.nvidia.com/nim/reference/openai-gpt-oss-20b-infer).
Also see [Nemotron Ultra](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b)
and [Cloud Functions invocation errors](https://docs.nvidia.com/nvcf/overview/generic-http-function-invocation).

## Leonida's identity

The Greek reply assistant's system prompt (`GREEK_SYSTEM` in `pipeline.py`)
introduces it as Leonida, a digital assistant for chatting in Greek and Tsakonian
that is here to help the user speak Tsakonian. It knows that the user writes in
Tsakonian, that messages reach it translated into Greek and that its reply is
translated back, so it asks for clarification when a translation makes no sense.
Leonida's translation prompts are unchanged.

## Interface language and starter phrases

The sidebar's **Language** control switches the interface between English, Greek
and Tsakonian. All interface texts and the starter phrases live in
`assets/i18n.json`; a missing Tsakonian text falls back to English. The Tsakonian
texts are translated once from the Greek ones by Leonida, then reviewed by hand:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/translate_ui.py
```

The script fills only empty `tsd` values, so reviewed corrections are never
overwritten; clear a value to translate it again. When a conversation is empty,
Leonida greets the user and offers the starter phrases as buttons; clicking one
sends it with its stored Greek (`ell`), so the first translation step is skipped. Starters without a Tsakonian text are hidden.

## Activity log

Every session writes to the console and to `logs/activity.log` (1 MB files, 3 kept,
ignored by Git). It records, per session id:

- each message the user types or starter they click;
- each stage with its model, duration, input text and result: Tsakonian → Greek
  (Leonida), the Greek reply (Ollama or NIM) and Greek → Tsakonian (Leonida);
- skipped stages, failures, retries, discarded turns and cleared conversations.

The log holds conversation text, so keep it private. Set `LOG_LEVEL` (for example
`WARNING`) in `.env` to log less.

## Appearance

The theme follows the Tsakonian Digital Dictionary: a grey page, white
cards, dark/royal blue accents, Calibri/Segoe UI typography, and matching navigation.
Messages are conversation bubbles: the user's on the right with the brand gradient
and white text, Leonida's on the left in white with the helmet avatar. The message
box and a separate round send button are pinned at the bottom; Enter also sends.
`assets/logo.png` and `assets/favicon.png` are copies of the dictionary's original
brand assets. The helmet is used as the browser favicon and assistant avatar.
`ui.py` and `assets/style.css` hold the presentation; `.streamlit/config.toml`
defines matching light and dark themes. The app follows the system setting or the
theme chosen in Streamlit's menu, and `style.css` switches its colour variables to
match. Both mobile and desktop layouts are supported.

## Checks

```powershell
.venv/Scripts/python.exe -m unittest -v test_lab
```

These tests cover the exact bidirectional prompts and single model name,
keep-alive, Greek history, rejected empty/truncated replies, and the Streamlit
chat flow including retries at each stage, repeated failures, reuse of completed
results, discarding a pending message, and conversation clearing. Network calls
are mocked, so these tests need neither model access nor API credits.
They also cover the reasoning budget, bounded output-limit recovery, and branded
asset loading. Desktop/mobile browser checks verify the favicon and page layout.
Local-backend tests cover Greek chat history, no API-key requirement, reuse of
the same Leonida model, incomplete answers, and switching a failed cloud reply
to KriKri without rerunning the source translation.
Message checks verify translation placement for both speakers, persistence after
retry, HTML escaping, and removal of completed pipeline details. Sidebar checks
cover the installed-model list, the NIM model list, a missing NVIDIA key, blank
messages and switching the interface language without losing the conversation.

Initial live validation on 2026-10-21 passed: a complete Streamlit chat turn took
28.8 seconds, translating `Κα ένι, ευχαριστού.` to `Καλά είμαι, ευχαριστώ.`,
obtaining a Greek NIM reply, and translating it back into Tsakonian. `ollama ps`
showed only `leonida` loaded (CPU, 4096-token context, one-hour keep-alive).

After the output-limit fix later that day, all 13 automated checks passed.
Browser checks at 1280px and 390px verified the responsive layout and confirmed
that the served favicon exactly matches the dictionary's PNG. Live NIM generation
was timing out, including direct HTTP and streaming requests for a simple greeting,
while model listing worked. The revised live pipeline could therefore not be
verified during that check; timeout recovery preserves the translated input.

Subsequent checks isolated HTTP 504 failures to the former GPT-OSS route and
found working replacement models. Switching to Nemotron Ultra allowed a live
retry of the saved Leonidio question to complete, including its Tsakonian output,
in 50.6 seconds. Only the final local translation ran during that retry.

The optional local KriKri recovery path also completed live in 101 seconds,
using only two local requests: one to KriKri for the Greek reply and one to
Leonida for its Tsakonian translation. No NVIDIA client was used. All 19 automated
checks passed after the provider selector and user-translation display changes.

The current default is local Google Gemma 4 E4B. Live Greek checks completed in
26.2 seconds on the first request and 13.5–15.5 seconds on subsequent requests.
These are small conversational checks, not a Greek accuracy benchmark.
A complete browser chat turn using Gemma and the same Leonida translator took
57.2 seconds. All 20 automated checks pass. Desktop/mobile checks measured the
same 4.16-pixel text-to-Greek gap for both speakers and confirmed that completed
messages have no pipeline expander or progress panel.
