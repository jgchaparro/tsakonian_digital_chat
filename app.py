import importlib
import os
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

from pipeline import DEFAULT_NIM_MODEL, DEFAULT_OLLAMA_MODEL, KRIKRI_MODEL, ask_nim, ask_ollama, translate
from ui import FAVICON, render_empty_state, render_footer, render_header

env_path = Path(__file__).with_name(".env")
load_dotenv(env_path)

# Read the local model choice on every rerun. load_dotenv previously copied the
# old choice into os.environ, which otherwise kept it after editing .env.
local_settings = dotenv_values(env_path)
NIM_MODEL = local_settings.get("NIM_MODEL") or os.environ.get("NIM_MODEL", DEFAULT_NIM_MODEL)
OLLAMA_CHAT_MODEL = local_settings.get("OLLAMA_CHAT_MODEL") or os.environ.get("OLLAMA_CHAT_MODEL", DEFAULT_OLLAMA_MODEL)
OLLAMA_CHAT_KEEP_ALIVE = os.environ.get("OLLAMA_CHAT_KEEP_ALIVE", os.environ.get("KRIKRI_KEEP_ALIVE", "5m"))
LEONIDA_MODEL = os.environ.get("LEONIDA_MODEL", "leonida")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "1h")
NIM_MAX_TOKENS = int(os.environ.get("NIM_MAX_TOKENS", "2048"))
NIM_REASONING_EFFORT = os.environ.get("NIM_REASONING_EFFORT", "low")

st.set_page_config(page_title="Tsakonian Digital Lab", page_icon=FAVICON)
render_header()

api_key = os.environ.get("NVIDIA_API_KEY")
with st.bottom:
    with st.container(key="composer", horizontal=True, wrap=False,
                      vertical_alignment="center", gap="small"):
        with st.popover("Assistant settings", icon=":material/tune:", type="primary",
                        width=48, help="Choose the Greek reply assistant"):
            st.markdown("**Greek reply assistant**")
            default_backend = local_settings.get("GREEK_BACKEND") or os.environ.get("GREEK_BACKEND", "ollama")
            # Apply edited defaults to open sessions once, while retaining later
            # choices made with the settings controls and all conversation state.
            defaults = (default_backend, OLLAMA_CHAT_MODEL)
            if st.session_state.get("assistant_defaults") != defaults:
                st.session_state.greek_backend = default_backend
                st.session_state.local_greek_model = OLLAMA_CHAT_MODEL
                st.session_state.assistant_defaults = defaults
            backend = st.selectbox(
                "Provider", ["nim", "ollama"],
                format_func=lambda value: "NVIDIA NIM" if value == "nim" else "Local Ollama",
                key="greek_backend",
            )
            if backend == "ollama":
                local_models = list(dict.fromkeys([OLLAMA_CHAT_MODEL, DEFAULT_OLLAMA_MODEL, KRIKRI_MODEL]))
                OLLAMA_CHAT_MODEL = st.selectbox(
                    "Local model", local_models, key="local_greek_model",
                    format_func=lambda value: {
                        DEFAULT_OLLAMA_MODEL: "Google Gemma 4 E4B",
                        KRIKRI_MODEL: "ILSP KriKri 8B (Greek)",
                    }.get(value, value),
                )
                st.caption(
                    "Greek replies stay on this computer. "
                    "Ollama may reload models between steps when memory is limited."
                )
            else:
                st.caption(f"Greek replies use {NIM_MODEL} on NVIDIA NIM.")
        prompt = st.chat_input(
            "Write in Tsakonian…",
            disabled=st.session_state.get("pending_turn") is not None or (backend == "nim" and not api_key),
        )

if backend == "nim" and not api_key:
    st.error("Missing NVIDIA_API_KEY. Add it to a .env file or to your environment.")
    st.stop()

nim = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", api_key=api_key,
    timeout=60, max_retries=0,
) if backend == "nim" else None


def leonida(text, origin, target):
    return translate(
        text, origin, target, model=LEONIDA_MODEL,
        base_url=OLLAMA_BASE_URL, keep_alive=OLLAMA_KEEP_ALIVE,
    )


def show_turn(turn):
    show_user_message(turn)
    with st.chat_message("assistant", avatar=FAVICON):
        show_message_text(turn["tsd_out"], turn["ell_out"], role="assistant")


def show_greek_translation(slot, text, role="user"):
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
    with st.chat_message("user", avatar=":material/person:"):
        return show_message_text(turn["tsd_in"], turn.get("ell_in"), role="user")


if "turns" not in st.session_state:
    st.session_state.turns = []
if "pending_turn" not in st.session_state:
    st.session_state.pending_turn = None

conversation_column, clear_column = st.columns([3, 1])
conversation_column.markdown("**Conversation**")
if clear_column.button("Clear conversation", use_container_width=True):
    st.session_state.turns = []
    st.session_state.pending_turn = None
    st.rerun()

if prompt:
    st.session_state.pending_turn = {"tsd_in": prompt}

for turn in st.session_state.turns:
    show_turn(turn)

if not st.session_state.turns and st.session_state.pending_turn is None:
    render_empty_state()

pending = st.session_state.pending_turn
if pending is not None:
    greek_slot = show_user_message(pending)
    with st.chat_message("assistant", avatar=FAVICON):
        run_turn = "error" not in pending
        if "error" in pending:
            st.error(pending["error"])
            if "ell_in" in pending:
                with st.expander("Completed steps", expanded=True):
                    st.markdown(f"**1. Leonida (Tsakonian → Greek)**\n\n{pending['ell_in']}")
                    if "ell_out" in pending:
                        st.markdown(f"**2. {pending.get('greek_model', 'Assistant')} (Greek reply)**\n\n{pending['ell_out']}")
            retry_column, discard_column = st.columns(2)
            run_turn = retry_column.button("Retry failed step", type="primary")
            if discard_column.button("Discard message"):
                st.session_state.pending_turn = None
                st.rerun()

        if run_turn:
            pending.pop("error", None)
            stage = "Tsakonian → Greek"
            try:
                # Save each result immediately. A retry skips completed stages.
                with st.status("Running pipeline…", expanded=True) as status:
                    if "ell_in" not in pending:
                        st.write("Leonida: Tsakonian → Greek")
                        pending["ell_in"] = leonida(pending["tsd_in"], "Tsakonian", "Greek")
                        show_greek_translation(greek_slot, pending["ell_in"])
                    st.write(f"Greek: {pending['ell_in']}")
                    stage = "Greek reply"
                    if "ell_out" not in pending:
                        if backend == "ollama":
                            st.write(f"{OLLAMA_CHAT_MODEL}: generating Greek reply")
                            pending["ell_out"] = ask_ollama(
                                st.session_state.turns, pending["ell_in"], model=OLLAMA_CHAT_MODEL,
                                base_url=OLLAMA_BASE_URL, keep_alive=OLLAMA_CHAT_KEEP_ALIVE,
                            )
                            pending["greek_model"] = OLLAMA_CHAT_MODEL
                        else:
                            st.write(f"{NIM_MODEL}: generating reply")
                            pending["ell_out"] = ask_nim(
                                nim, NIM_MODEL, st.session_state.turns, pending["ell_in"],
                                max_tokens=NIM_MAX_TOKENS, reasoning_effort=NIM_REASONING_EFFORT,
                            )
                            pending["greek_model"] = NIM_MODEL
                    st.write(f"Greek reply: {pending['ell_out']}")
                    stage = "Greek → Tsakonian"
                    if "tsd_out" not in pending:
                        st.write("Leonida: Greek → Tsakonian")
                        pending["tsd_out"] = leonida(pending["ell_out"], "Greek", "Tsakonian")
                    status.update(label="Done", state="complete", expanded=False)
            except APITimeoutError:
                pending["error"] = (
                    "The Greek reply service did not respond in time. "
                    "Your completed steps are saved. Please retry this step."
                )
                st.rerun()
            except Exception as e:
                pending["error"] = f"{stage} failed: {e}"
                st.rerun()
            st.session_state.turns.append(pending.copy())
            st.session_state.pending_turn = None
            st.rerun()

render_footer()
