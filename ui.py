"""Shared Tsakonian Digital branding, using the dictionary's original assets."""

import base64
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).with_name("assets")
FAVICON = str(ASSETS / "favicon.png")


def render_header():
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    logo = base64.b64encode((ASSETS / "logo.png").read_bytes()).decode("ascii")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown(
        f'''<header class="lab-header">
          <a href="https://tsakoniandigital.com/" target="_blank" rel="noopener noreferrer"
             aria-label="Tsakonian Digital home">
            <img src="data:image/png;base64,{logo}" alt="Tsakonian Digital" />
          </a>
          <nav aria-label="Main navigation">
            <a href="https://tsakoniandigital.com/" target="_blank" rel="noopener noreferrer">Dictionary</a>
            <span aria-current="page">Digital Lab</span>
            <a href="https://tsakoniandigital.com/keyboard" target="_blank" rel="noopener noreferrer">Writing helper</a>
          </nav>
        </header>''',
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
        '''<section class="lab-empty">
          <div class="lab-eyebrow">YOUR CONVERSATION STARTS HERE</div>
          <h2>Write your first message</h2>
          <p>Use the message box below to begin. Greek text appears directly
             beneath each message.</p>
        </section>''',
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        '<footer class="lab-footer">Built by Jaime García Chaparro for the Tsakonian Digital project</footer>',
        unsafe_allow_html=True,
    )
