"""Centralised UI theme.

Holds the global CSS, the color palette and a single :func:`apply_theme`
helper that every Streamlit page calls right after ``st.set_page_config``.
The visual language is a quiet, professional slate-dark SaaS aesthetic:
solid backgrounds, hairline borders, a single indigo accent, restrained
hover transitions.
"""

from __future__ import annotations

import streamlit as st


PALETTE = {
    "bg": "#0b0f17",
    "bg_elevated": "#11161f",
    "bg_elevated_2": "#161b26",
    "bg_input": "#0e131c",
    "border": "#232a38",
    "border_strong": "#2e3648",
    "text": "#e6edf3",
    "text_muted": "#9aa4b2",
    "text_dim": "#6b7384",
    "primary": "#6366f1",
    "primary_hover": "#4f46e5",
    "primary_soft": "rgba(99,102,241,0.12)",
    "success": "#4ade80",
    "warning": "#f59e0b",
    "danger": "#ef4444",
}


_GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0b0f17;
    --bg-elevated: #11161f;
    --bg-elevated-2: #161b26;
    --bg-input: #0e131c;
    --border: #232a38;
    --border-strong: #2e3648;
    --text: #e6edf3;
    --text-muted: #9aa4b2;
    --text-dim: #6b7384;
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --primary-soft: rgba(99,102,241,0.12);
    --primary-soft-strong: rgba(99,102,241,0.20);
    --success: #4ade80;
    --warning: #f59e0b;
    --danger: #ef4444;
    --focus-ring: rgba(99,102,241,0.35);
    --radius: 10px;
    --radius-sm: 6px;
    --radius-lg: 14px;
    --duration: 140ms;
}

/* ---------- Typography ---------- */

/* Apply Inter to text only. Critically, exclude Material Symbols icon
   spans - Streamlit relies on the icon font glyphs and forcing Inter
   onto them prints the icon name as literal text (e.g. "arrow_right"). */
body, p, span, div, label, button, input, textarea, select,
h1, h2, h3, h4, h5, h6,
.stMarkdown, .stText, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, Helvetica, Arial, sans-serif;
}

span[data-testid="stIconMaterial"],
span.material-symbols-rounded,
span.material-symbols-outlined,
span.material-icons,
span.material-icons-outlined,
[class*="material-symbols"],
[class*="material-icons"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Icons', 'Material Icons Outlined' !important;
    font-feature-settings: 'liga';
    -webkit-font-feature-settings: 'liga';
    text-rendering: optimizeLegibility;
}

body {
    color: var(--text);
    font-size: 14px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ---------- App shell ---------- */

.stApp {
    background: var(--bg) !important;
    min-height: 100vh;
}

/* Keep the Streamlit header alive (it hosts the sidebar expand button
   when the sidebar is collapsed). Just make it visually quiet. */
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
}

/* When the sidebar is collapsed, the expand control sits inside the
   header. Make sure it's clearly visible and styled like our nav. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    transition: background-color var(--duration), border-color var(--duration) !important;
}

[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="collapsedControl"]:hover {
    background: var(--bg-elevated-2) !important;
    border-color: var(--border-strong) !important;
}

[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {
    fill: var(--text) !important;
    color: var(--text) !important;
}

.main .block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 4rem !important;
    max-width: 1180px;
}

/* ---------- Headings ---------- */

h1, h2, h3, h4, h5 {
    color: var(--text);
    letter-spacing: -0.01em;
    font-weight: 600;
    margin: 0;
}

h1 { font-size: 1.6rem; font-weight: 700; line-height: 1.25; }
h2 { font-size: 1.2rem; line-height: 1.3; margin-top: 1.2rem; }
h3 { font-size: 1.02rem; line-height: 1.35; margin-top: 1rem; }

p { color: var(--text); margin: 0 0 0.6rem 0; }
a { color: var(--primary); text-decoration: none; }
a:hover { color: var(--primary-hover); text-decoration: underline; }

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {
    background: var(--bg-elevated-2) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--text);
    font-weight: 600;
    letter-spacing: -0.005em;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: var(--text);
}

/* Multi-page nav */
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
    padding: 0.4rem 0.5rem 1rem 0.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.8rem;
}

section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
    border-radius: var(--radius-sm) !important;
    margin: 1px 4px !important;
    padding: 7px 12px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: background-color var(--duration), color var(--duration);
    border-left: 2px solid transparent;
}

section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
    background: var(--bg-elevated) !important;
    color: var(--text) !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
    background: var(--primary-soft) !important;
    color: var(--text) !important;
    border-left: 2px solid var(--primary);
}

/* ---------- Buttons ---------- */

.stButton > button,
.stDownloadButton > button {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0 !important;
    transition: background-color var(--duration), border-color var(--duration), color var(--duration) !important;
    box-shadow: none !important;
    line-height: 1.4 !important;
    min-height: 38px !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: var(--bg-elevated) !important;
    border-color: var(--border-strong) !important;
    color: var(--text) !important;
    transform: none !important;
}

.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 3px var(--focus-ring) !important;
    border-color: var(--primary) !important;
}

.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: #ffffff !important;
    border-color: var(--primary) !important;
}

.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
    background: var(--primary-hover) !important;
    border-color: var(--primary-hover) !important;
    color: #ffffff !important;
}

/* Form submit buttons */
.stForm button[kind="primaryFormSubmit"],
.stForm button[kind="secondaryFormSubmit"] {
    border-radius: var(--radius) !important;
}

/* ---------- Inputs ---------- */

.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    transition: border-color var(--duration), box-shadow var(--duration) !important;
    font-size: 0.9rem !important;
}

.stTextArea textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    transition: border-color var(--duration), box-shadow var(--duration);
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.55 !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus,
.stDateInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--focus-ring) !important;
    outline: none !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--focus-ring) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-size: 0.92rem !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stChatInput"] button {
    background: var(--primary) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    transition: background-color var(--duration);
}

[data-testid="stChatInput"] button:hover {
    background: var(--primary-hover) !important;
}

/* ---------- Toggle / Switch ---------- */

/* Streamlit's default checked toggle is red; restyle to indigo. */
[data-testid="stToggle"] label > div:first-child,
.stCheckbox label > div:first-child,
div[data-baseweb="checkbox"] > div:first-child {
    background-color: var(--bg-elevated) !important;
    border-color: var(--border-strong) !important;
}

[data-testid="stToggle"] input:checked ~ label > div:first-child,
[data-testid="stToggle"] label[data-checked="true"] > div:first-child {
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
}

/* baseweb switch (used by st.toggle) */
[role="switch"][aria-checked="true"] {
    background-color: var(--primary) !important;
}
[role="switch"][aria-checked="true"] > div {
    background-color: #ffffff !important;
}
[role="switch"][aria-checked="false"] {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-strong) !important;
}
[role="switch"][aria-checked="false"] > div {
    background-color: var(--text-dim) !important;
}

/* ---------- Chat messages ---------- */

[data-testid="stChatMessage"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.85rem 1.05rem !important;
    margin-bottom: 0.7rem !important;
    animation: mtFadeIn 140ms ease-out;
}

[data-testid="stChatMessage"][data-testid*="user"] {
    background: var(--primary-soft) !important;
    border-color: rgba(99,102,241,0.28) !important;
    border-left: 3px solid var(--primary) !important;
}

@keyframes mtFadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

/* ---------- Expanders ---------- */

[data-testid="stExpander"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
    transition: border-color var(--duration);
}

[data-testid="stExpander"]:hover {
    border-color: var(--border-strong) !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > summary {
    padding: 0.6rem 0.9rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    font-size: 0.88rem !important;
}

[data-testid="stExpander"] summary:hover {
    color: var(--text) !important;
}

/* ---------- Code blocks ---------- */

.stCodeBlock, pre {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.82rem !important;
}

code {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
}

:not(pre) > code {
    background: var(--bg-elevated) !important;
    color: var(--text) !important;
    padding: 1px 6px !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    font-size: 0.82em !important;
}

/* ---------- Captions ---------- */

.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-dim) !important;
    font-size: 0.8rem !important;
}

/* ---------- Tabs ---------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin-bottom: 1.2rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 0 !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 10px 16px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin: 0 !important;
    transition: color var(--duration), border-color var(--duration);
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text) !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: transparent !important;
    color: var(--text) !important;
    border-bottom: 2px solid var(--primary) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* ---------- Alerts ---------- */

.stAlert {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-elevated) !important;
}

div[data-baseweb="notification"] {
    border-radius: var(--radius) !important;
}

/* ---------- Dividers ---------- */

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.4rem 0 !important;
}

/* ---------- Metrics ---------- */

[data-testid="stMetric"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.85rem 1rem !important;
    transition: border-color var(--duration);
}

[data-testid="stMetric"]:hover {
    border-color: var(--border-strong) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-dim) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}

/* ---------- File uploader ---------- */

[data-testid="stFileUploader"] section {
    background: var(--bg-elevated) !important;
    border: 1px dashed var(--border-strong) !important;
    border-radius: var(--radius) !important;
}

[data-testid="stFileUploader"] section:hover {
    border-color: var(--primary) !important;
}

/* ---------- Radio / Selectbox ---------- */

.stRadio > div {
    gap: 0.4rem;
}

.stRadio label {
    color: var(--text) !important;
}

/* ---------- Hide Streamlit chrome ---------- */

/* Only hide the hamburger menu and the "Made with Streamlit" footer.
   Do NOT hide the toolbar wholesale - the sidebar-expand button when
   the sidebar is collapsed lives inside the toolbar/header region. */
#MainMenu,
footer,
[data-testid="stStatusWidget"],
[data-testid="stDeployButton"] {
    display: none !important;
}

/* ---------- Custom helper classes ---------- */

.mt-hero {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem 1.5rem 1.6rem;
    margin-bottom: 1.4rem;
}

.mt-hero-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem 0;
    line-height: 1.25;
}

.mt-hero-subtitle {
    color: var(--text-muted);
    font-size: 0.92rem;
    line-height: 1.55;
    max-width: 760px;
    margin: 0;
}

.mt-hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 0.9rem;
}

.mt-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 2px 9px;
    background: var(--bg-elevated-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 500;
    color: var(--text-muted);
    line-height: 1.6;
    white-space: nowrap;
}

.mt-badge-primary {
    background: var(--primary-soft);
    border-color: rgba(99,102,241,0.32);
    color: #c7d2fe;
}

.mt-badge-success {
    background: rgba(74,222,128,0.10);
    border-color: rgba(74,222,128,0.30);
    color: #86efac;
}

.mt-badge-warning {
    background: rgba(245,158,11,0.10);
    border-color: rgba(245,158,11,0.30);
    color: #fcd34d;
}

.mt-badge-danger {
    background: rgba(239,68,68,0.10);
    border-color: rgba(239,68,68,0.30);
    color: #fca5a5;
}

.mt-badge-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-dim);
}

.mt-badge-success .mt-badge-dot { background: var(--success); }
.mt-badge-warning .mt-badge-dot { background: var(--warning); }
.mt-badge-danger  .mt-badge-dot { background: var(--danger); }
.mt-badge-primary .mt-badge-dot { background: var(--primary); }

.mt-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.95rem 1.1rem;
    margin-bottom: 0.7rem;
    transition: border-color var(--duration);
}

.mt-card:hover {
    border-color: var(--border-strong);
}

.mt-card-title {
    font-size: 0.96rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 0.25rem 0;
}

.mt-card-meta {
    font-size: 0.78rem;
    color: var(--text-dim);
    margin: 0 0 0.5rem 0;
}

.mt-feature-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.25rem;
    transition: border-color var(--duration);
    height: 100%;
}

.mt-feature-card:hover {
    border-color: var(--border-strong);
}

.mt-feature-icon {
    width: 32px;
    height: 32px;
    border-radius: var(--radius-sm);
    background: var(--primary-soft);
    color: var(--primary);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    margin-bottom: 0.7rem;
    font-weight: 600;
}

.mt-feature-title {
    font-size: 0.98rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 0.35rem 0;
}

.mt-feature-desc {
    font-size: 0.86rem;
    color: var(--text-muted);
    line-height: 1.6;
    margin: 0;
}

.mt-stat-row {
    display: flex;
    gap: 1.1rem;
    flex-wrap: wrap;
    margin-top: 0.25rem;
}

.mt-stat {
    color: var(--text-muted);
    font-size: 0.8rem;
}

.mt-stat strong {
    color: var(--text);
    font-weight: 600;
    margin-right: 4px;
}

.mt-subtle-divider {
    color: var(--text-dim);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.5rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

.mt-source-card {
    background: var(--bg-elevated-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.5rem;
}

.mt-source-head {
    display: flex;
    justify-content: space-between;
    gap: 0.6rem;
    align-items: baseline;
}

.mt-source-title {
    font-weight: 600;
    color: var(--text);
    font-size: 0.86rem;
}

.mt-source-meta {
    color: var(--text-dim);
    font-size: 0.74rem;
}

.mt-source-speakers {
    color: var(--text-muted);
    font-size: 0.76rem;
    margin-top: 2px;
}

.mt-mini-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0.2rem;
    border-bottom: 1px solid var(--border);
}

.mt-mini-stat:last-child {
    border-bottom: none;
}

.mt-mini-stat-label {
    color: var(--text-muted);
    font-size: 0.82rem;
}

.mt-mini-stat-value {
    color: var(--text);
    font-weight: 600;
    font-size: 0.92rem;
    font-feature-settings: 'tnum';
}

.mt-sidebar-section {
    margin: 0.4rem 0 1rem 0;
}

.mt-sidebar-title {
    color: var(--text);
    font-weight: 600;
    font-size: 1rem;
    margin: 0;
    letter-spacing: -0.01em;
}

.mt-sidebar-subtitle {
    color: var(--text-dim);
    font-size: 0.78rem;
    margin: 0.2rem 0 0 0;
}

.mt-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text);
    font-weight: 600;
    font-size: 1.02rem;
    letter-spacing: -0.01em;
    margin: 0;
}

.mt-brand-mark {
    width: 22px;
    height: 22px;
    border-radius: 6px;
    background: var(--primary);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 0.7rem;
    font-weight: 700;
}

/* Danger zone button (admin) - on hover, shifts to red border */
.mt-danger-zone .stButton > button:hover {
    border-color: var(--danger) !important;
    color: #fca5a5 !important;
    background: rgba(239,68,68,0.06) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--border-strong);
    border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* Spinner */
.stSpinner > div { border-top-color: var(--primary) !important; }
"""


def apply_theme(page_title: str = "Multi-Transcript Q&A", page_icon: str = "\U0001F4DA") -> None:
    """Apply the global theme. Call once per page right after page_config."""
    st.markdown(f"<style>{_GLOBAL_CSS}</style>", unsafe_allow_html=True)


__all__ = ["PALETTE", "apply_theme"]
