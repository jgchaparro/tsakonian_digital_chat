import importlib
import logging
import os
import time
import uuid
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values, load_dotenv
from openai import APITimeoutError, OpenAI

import pipeline as pipeline_module
import ui as ui_module


@st.cache_resource(show_spinner=False)
def load_helpers(pipeline_version, ui_version):
    # The Windows watcher can miss helper edits in this accented workspace path.
    # Reload changed helpers without restarting and losing conversation state.
    return importlib.reload(pipeline_module), importlib.reload(ui_module)


load_helpers(Path(pipeline_module.__file__).stat().st_mtime_ns,
             Path(ui_module.__file__).stat().st_mtime_ns)

from pipeline import (DEFAULT_NIM_MODEL, DEFAULT_OLLAMA_MODEL, KRIKRI_MODEL, ask_nim, ask_ollama,
                      list_ollama_models, parse_model_list, translate)
from activity_log import clip, log, setup_logging
from ui import FAVICON, caps, load_texts, render_footer, render_header, render_welcome, translator

env_path = Path(__file__).with_name(".env")
load_dotenv(env_path)

# Read the local model choice on every rerun. load_dotenv previously copied the
# old choice into os.environ, which otherwise kept it after editing .env.
local_settings = dotenv_values(env_path)
NIM_MODEL = local_settings.get("NIM_MODEL") or os.environ.get("NIM_MODEL", DEFAULT_NIM_MODEL)
NIM_MODELS = parse_model_list(local_settings.get("NIM_MODELS") or os.environ.get("NIM_MODELS"), NIM_MODEL)
OLLAMA_CHAT_MODEL = local_settings.get("OLLAMA_CHAT_MODEL") or os.environ.get("OLLAMA_CHAT_MODEL", DEFAULT_OLLAMA_MODEL)
OLLAMA_CHAT_KEEP_ALIVE = os.environ.get("OLLAMA_CHAT_KEEP_ALIVE", os.environ.get("KRIKRI_KEEP_ALIVE", "5m"))
LEONIDA_MODEL = os.environ.get("LEONIDA_MODEL", "leonida")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "1h")
NIM_MAX_TOKENS = int(os.environ.get("NIM_MAX_TOKENS", "2048"))
NIM_REASONING_EFFORT = os.environ.get("NIM_REASONING_EFFORT", "low")
MODEL_LABELS = {DEFAULT_OLLAMA_MODEL: "Google Gemma 4 E4B", KRIKRI_MODEL: "ILSP KriKri 8B (Greek)"}


@st.cache_data(ttl=30, show_spinner=False)
def installed_ollama_models(base_url, translator_model):
    return list_ollama_models(base_url, exclude=("leonida", translator_model.split(":")[0]))


st.set_page_config(page_title="Tsakonian Digital Chat", page_icon=FAVICON)
texts = load_texts()
setup_logging()
if "log_id" not in st.session_state:
    st.session_state.log_id = uuid.uuid4().hex[:6]
    log.info("[%s] session started", st.session_state.log_id)


def note(message, level=logging.INFO):
    log.log(level, "[%s] %s", st.session_state.log_id, message)


if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "tsd"
if "show_greek" not in st.session_state:
    st.session_state.show_greek = True
t = translator(texts, st.session_state.ui_lang)
render_header(t, st.context.theme.type)

api_key = os.environ.get("NVIDIA_API_KEY")
with st.sidebar:
    st.segmented_control(
        caps(t("language")), list(texts["languages"]), key="ui_lang", required=True,
        format_func=texts["languages"].get, width="stretch",
    )
    default_backend = local_settings.get("GREEK_BACKEND") or os.environ.get("GREEK_BACKEND", "ollama")
    # Apply edited defaults to open sessions once, while retaining later
    # choices made with the settings controls and all conversation state.
    defaults = (default_backend, OLLAMA_CHAT_MODEL, NIM_MODEL)
    if st.session_state.get("assistant_defaults") != defaults:
        st.session_state.greek_backend = default_backend
        st.session_state.local_greek_model = OLLAMA_CHAT_MODEL
        st.session_state.nim_model = NIM_MODEL
        st.session_state.assistant_defaults = defaults
    backend = st.segmented_control(
        caps(t("assistant")), ["ollama", "nim"], key="greek_backend", required=True,
        format_func=lambda value: "NVIDIA NIM" if value == "nim" else "Ollama", width="stretch",
    )
    if backend == "ollama":
        local_models = list(dict.fromkeys([
            OLLAMA_CHAT_MODEL, DEFAULT_OLLAMA_MODEL, KRIKRI_MODEL,
            *installed_ollama_models(OLLAMA_BASE_URL, LEONIDA_MODEL),
        ]))
        OLLAMA_CHAT_MODEL = st.selectbox(
            caps(t("model")), local_models, key="local_greek_model",
            format_func=lambda value: MODEL_LABELS.get(value, value),
        )
        st.caption(t("local_caption"))
    else:
        NIM_MODEL = st.selectbox(caps(t("model")), NIM_MODELS, key="nim_model")
        st.caption(t("nim_caption"))
        if not api_key:
            st.warning(t("missing_key"))
    st.toggle(caps(t("show_greek")), key="show_greek")
    st.divider()
    if st.button(t("clear"), icon=":material/delete:", width="stretch"):
        note(f"conversation cleared ({len(st.session_state.get('turns', []))} turns)")
        st.session_state.turns = []
        st.session_state.pending_turn = None
        st.rerun()

show_greek = st.session_state.show_greek
locked = st.session_state.get("pending_turn") is not None or (backend == "nim" and not api_key)
with st.bottom:
    with st.form("composer", clear_on_submit=True, border=False):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            message = st.text_input(t("placeholder"), key="message", placeholder=t("placeholder"),
                                    label_visibility="collapsed", disabled=locked)
            sent = st.form_submit_button(t("send"), key="send", type="primary",
                                         icon=":material/send:", disabled=locked)
prompt = message.strip() if sent and message else None

nim = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", api_key=api_key,
    timeout=60, max_retries=0,
) if backend == "nim" and api_key else None


def leonida(text, origin, target):
    return translate(
        text, origin, target, model=LEONIDA_MODEL,
        base_url=OLLAMA_BASE_URL, keep_alive=OLLAMA_KEEP_ALIVE,
    )


def timed(stage, model, source, call):
    """Run one pipeline stage, logging its input, duration and result or failure."""
    note(f"{stage} [{model}] started: {clip(source)}")
    started = time.perf_counter()
    try:
        result = call()
    except Exception as e:
        note(f"{stage} [{model}] FAILED after {time.perf_counter() - started:.1f}s: {type(e).__name__}: {e}",
             logging.ERROR)
        raise
    note(f"{stage} [{model}] done in {time.perf_counter() - started:.1f}s: {clip(result)}")
    return result


def show_turn(turn):
    show_user_message(turn)
    with st.chat_message("assistant", avatar=FAVICON):
        show_message_text(turn["tsd_out"], turn["ell_out"], role="assistant")


def show_greek_translation(slot, text, role="user"):
    if show_greek:
        slot.markdown(
            f'<div class="{role}-greek-translation" lang="el">{escape(text)}</div>',
            unsafe_allow_html=True,
        )


def show_message_text(text, greek_text=None, *, role):
    # Both speakers use the same text layout and translation spacing.
    with st.container(gap=None):
        st.write(text)
        greek_slot = st.empty()
        if greek_text is not None:
            show_greek_translation(greek_slot, greek_text, role=role)
    return greek_slot


def show_user_message(turn):
    with st.chat_message("user"):
        return show_message_text(turn["tsd_in"], turn.get("ell_in"), role="user")


if "turns" not in st.session_state:
    st.session_state.turns = []
if "pending_turn" not in st.session_state:
    st.session_state.pending_turn = None

if prompt:
    note(f"user message: {clip(prompt)}")
    st.session_state.pending_turn = {"tsd_in": prompt}

for turn in st.session_state.turns:
    show_turn(turn)

if not st.session_state.turns and st.session_state.pending_turn is None:
    starter = render_welcome(texts, t, st.session_state.ui_lang, show_greek)
    if starter:
        # The stored Greek is the exact source of the phrase, so it needs no translating.
        note(f"starter clicked: {clip(starter['tsd'])}")
        st.session_state.pending_turn = {"tsd_in": starter["tsd"], "ell_in": starter["ell"]}
        st.rerun()

pending = st.session_state.pending_turn
if pending is not None:
    greek_slot = show_user_message(pending)
    with st.chat_message("assistant", avatar=FAVICON):
        run_turn = "error" not in pending
        if "error" in pending:
            st.error(pending["error"])
            if "ell_in" in pending:
                with st.expander(t("completed_steps"), expanded=True):
                    st.markdown(f"**1. Leonida (Tsakonian → Greek)**\n\n{pending['ell_in']}")
                    if "ell_out" in pending:
                        st.markdown(f"**2. {pending.get('greek_model', 'Assistant')} (Greek reply)**\n\n{pending['ell_out']}")
            retry_column, discard_column = st.columns(2)
            run_turn = retry_column.button(t("retry"), type="primary")
            if discard_column.button(t("discard")):
                note("failed turn discarded")
                st.session_state.pending_turn = None
                st.rerun()

        if run_turn:
            kept = [key for key in ("ell_in", "ell_out", "tsd_out") if key in pending]
            if "error" in pending:
                note(f"retry requested; completed stages kept: {', '.join(kept) or 'none'}")
            reply_model = NIM_MODEL if backend == "nim" else OLLAMA_CHAT_MODEL
            note(f"turn started (history {len(st.session_state.turns)} turns, reply assistant "
                 f"{backend}/{reply_model}, translator {LEONIDA_MODEL})")
            turn_started = time.perf_counter()
            pending.pop("error", None)
            stage = t("status_thinking" if "ell_in" in pending else "status_to_greek")
            try:
                # Save each result immediately. A retry skips completed stages.
                with st.status(stage, expanded=False) as status:
                    if "ell_in" not in pending:
                        pending["ell_in"] = timed(
                            "Tsakonian → Greek", LEONIDA_MODEL, pending["tsd_in"],
                            lambda: leonida(pending["tsd_in"], "Tsakonian", "Greek"))
                        show_greek_translation(greek_slot, pending["ell_in"])
                    else:
                        note(f"Tsakonian → Greek skipped, stored Greek: {clip(pending['ell_in'])}")
                    stage = t("status_thinking")
                    status.update(label=stage)
                    if "ell_out" not in pending:
                        if backend == "ollama":
                            pending["ell_out"] = timed(
                                "Greek reply", f"ollama/{OLLAMA_CHAT_MODEL}", pending["ell_in"],
                                lambda: ask_ollama(
                                    st.session_state.turns, pending["ell_in"], model=OLLAMA_CHAT_MODEL,
                                    base_url=OLLAMA_BASE_URL, keep_alive=OLLAMA_CHAT_KEEP_ALIVE,
                                ))
                            pending["greek_model"] = OLLAMA_CHAT_MODEL
                        else:
                            if nim is None:
                                raise ValueError(t("missing_key"))
                            pending["ell_out"] = timed(
                                "Greek reply", f"nim/{NIM_MODEL}", pending["ell_in"],
                                lambda: ask_nim(
                                    nim, NIM_MODEL, st.session_state.turns, pending["ell_in"],
                                    max_tokens=NIM_MAX_TOKENS, reasoning_effort=NIM_REASONING_EFFORT,
                                ))
                            pending["greek_model"] = NIM_MODEL
                    else:
                        note(f"Greek reply skipped, stored reply: {clip(pending['ell_out'])}")
                    stage = t("status_to_tsd")
                    status.update(label=stage)
                    if "tsd_out" not in pending:
                        pending["tsd_out"] = timed(
                            "Greek → Tsakonian", LEONIDA_MODEL, pending["ell_out"],
                            lambda: leonida(pending["ell_out"], "Greek", "Tsakonian"))
                    status.update(label=t("status_done"), state="complete")
            except APITimeoutError:
                note("turn stopped: the reply service timed out; completed stages are saved", logging.WARNING)
                pending["error"] = t("error_timeout")
                st.rerun()
            except Exception as e:
                note(f"turn stopped at '{stage.rstrip('…')}' ({type(e).__name__}: {e}); completed stages are saved", logging.WARNING)
                pending["error"] = f"{t('error_failed')} {stage.rstrip('…')}. {e}"
                st.rerun()
            note(f"turn complete in {time.perf_counter() - turn_started:.1f}s; Tsakonian reply: "
                 f"{clip(pending['tsd_out'])}")
            st.session_state.turns.append(pending.copy())
            st.session_state.pending_turn = None
            st.rerun()

render_footer(t)
