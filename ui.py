"""Shared Tsakonian Digital branding, using the dictionary's original assets."""

import base64
import json
import re
import unicodedata
from html import escape
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).with_name("assets")
FAVICON = str(ASSETS / "favicon.png")


def load_texts():
    return json.loads((ASSETS / "i18n.json").read_text(encoding="utf-8"))


def translator(texts, lang):
    # Untranslated interface texts fall back to English.
    def t(key):
        entry = texts["strings"][key]
        return entry.get(lang) or entry["en"]
    return t


def caps(text):
    """Upper-case a label the Greek way: capitals carry no accent.

    CSS text-transform keeps the accents, so labels are capitalised here. An accent
    that kept a vowel apart from a following ι/υ becomes a diaeresis (τσάι → ΤΣΑΪ).
    Tsakonian letter diacritics such as τ̔, σ̌ and ν̇ are kept.
    """
    text = unicodedata.normalize("NFD", text)
    text = re.sub("([αεοηΑΕΟΗ])\\u0301([ιυΙΥ])(?![\\u0301\\u0308])", "\\1\\2̈", text)
    text = text.replace("́", "")
    return unicodedata.normalize("NFC", text.upper())


def render_header(t, theme_type=None):
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    logo = base64.b64encode((ASSETS / "logo.png").read_bytes()).decode("ascii")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    # Streamlit's chosen theme can differ from the system one; the CSS reads this marker.
    theme = f'<span class="lab-theme-{theme_type}" hidden></span>' if theme_type else ""
    st.markdown(
        f'''{theme}<header class="lab-header">
          <a href="https://tsakoniandigital.com/" target="_blank" rel="noopener noreferrer"
             aria-label="Tsakonian Digital home">
            <img src="data:image/png;base64,{logo}" alt="Tsakonian Digital" />
          </a>
          <nav aria-label="Main navigation">
            <a href="https://tsakoniandigital.com/" target="_blank" rel="noopener noreferrer">{escape(t("nav_dictionary"))}</a>
            <span aria-current="page">{escape(t("nav_lab"))}</span>
            <a href="https://tsakoniandigital.com/keyboard" target="_blank" rel="noopener noreferrer">{escape(t("nav_writing"))}</a>
          </nav>
        </header>''',
        unsafe_allow_html=True,
    )


def render_welcome(texts, t, lang, show_greek):
    """Leonida's greeting with starter phrases; returns the starter clicked, if any."""
    with st.chat_message("assistant", avatar=FAVICON):
        # Same layout as a message bubble: when the first turn replaces this bubble,
        # Streamlit reuses its children, and any leftover sibling would show as stale text.
        with st.container(gap=None):
            st.write(t("welcome"))
            if lang == "tsd" and show_greek and texts["strings"]["welcome"].get("tsd"):
                st.markdown(
                    f'<div class="assistant-greek-translation" lang="el">'
                    f'{escape(texts["strings"]["welcome"]["el"])}</div>',
                    unsafe_allow_html=True,
                )
    # Starters appear once Leonida has translated them (scripts/translate_ui.py).
    starters = [s for s in texts["starters"] if s.get("tsd")]
    if not starters:
        return None
    with st.container(key="starters", horizontal=True, gap="small"):
        for i, starter in enumerate(starters):
            if st.button(starter["tsd"], key=f"starter_{i}",
                         help=starter["ell"] if show_greek else None):
                return starter
    return None


def render_footer(t):
    st.markdown(f'<footer class="lab-footer">{escape(t("footer"))}</footer>', unsafe_allow_html=True)
