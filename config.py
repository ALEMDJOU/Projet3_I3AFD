"""
config.py — Chargement des secrets (st.secrets ou .env) et injection du style global.
"""
import os
import streamlit as st
from dotenv import load_dotenv


def get_api_keys() -> dict:
    """
    Retourne les clés API.
    Priorité :
        1. st.secrets (Streamlit Cloud)
        2. fichier .env (développement local)
    """
    try:
        # En environnement Streamlit Cloud, st.secrets est disponible
        youtube_key = st.secrets["YOUTUBE_API_KEY"]
        gemini_key = st.secrets["GEMINI_API_KEY"]
        hf_key = st.secrets["HUGGINGFACE_API_KEY"]
    except (AttributeError, KeyError):
        # Fallback local : lecture depuis .env
        load_dotenv()
        youtube_key = os.getenv("YOUTUBE_API_KEY", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        hf_key = os.getenv("HUGGINGFACE_API_KEY", "")

    return {
        "youtube": youtube_key,
        "gemini": gemini_key,
        "huggingface": hf_key,
    }


def setup_page():
    """Configure la page Streamlit et injecte le CSS global AVA Pro."""
    st.set_page_config(
        page_title="AVA Pro - Agentic Video Analysis",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <link rel="stylesheet"
              href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        :root {
            --teal-primary: #00D4AC;
            --teal-deep:    #018EA9;
            --bg-light:     #F0F5F7;
            --sidebar-bg:   #F8FDFF;
            --white:        #FFFFFF;
            --text-dark:    #1E293B;
        }
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-light);
        }
        section[data-testid="stSidebar"] {
            background-color: var(--sidebar-bg) !important;
            border-right: 2px solid #E2E8F0;
        }
        .stTextInput > div > div > input {
            border-radius: 10px; border: 1px solid #CBD5E1;
            padding: 12px 15px; transition: all 0.3s;
        }
        .stTextInput > div > div > input:focus {
            border-color: var(--teal-primary);
            box-shadow: 0 0 0 3px rgba(0,212,172,0.1);
        }
        div.stButton > button:first-child {
            background: linear-gradient(135deg, var(--teal-primary) 0%, var(--teal-deep) 100%);
            color: white; border-radius: 12px; padding: 0.8rem 1.5rem;
            font-weight: 700; border: none; width: 100%;
            transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(0,212,172,0.3);
        }
        .info-card {
            background: var(--white); padding: 24px; border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0,123,255,0.05);
            border: 1px solid #E2E8F0; margin-bottom: 20px;
            transition: border-color 0.3s;
        }
        .info-card:hover { border-color: var(--teal-primary); }
        .section-title {
            color: var(--teal-deep); font-weight: 800; font-size: 1.6rem;
            margin-bottom: 1.2rem; display: flex; align-items: center; gap: 12px;
            border-left: 5px solid var(--teal-primary); padding-left: 15px;
        }
        .main-header {
            background: linear-gradient(135deg, var(--teal-primary) 0%, var(--teal-deep) 100%);
            padding: 40px; border-radius: 24px; color: white; margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0,212,172,0.2);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px; background-color: #E2EFF2;
            padding: 5px; border-radius: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px; background-color: transparent;
            border-radius: 10px; font-weight: 700; color: var(--text-dark);
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--white) !important;
            color: var(--teal-deep) !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }
        .step-item {
            display: flex; align-items: center; gap: 12px;
            padding: 10px 15px; border-radius: 10px;
            background: var(--white); margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        }
        .step-active { background: #E8F9F8; border: 1px solid var(--teal-primary); }
        .comment-box {
            background: white;
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid var(--teal-primary);
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            display: flex;
            gap: 15px;
            margin-top: 15px;
        }
        .analyst-avatar {
            width: 45px;
            height: 45px;
            background: linear-gradient(135deg, var(--teal-primary), var(--teal-deep));
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.2rem;
            flex-shrink: 0;
            box-shadow: 0 4px 8px rgba(0,212,172,0.2);
        }
        .comment-content { flex-grow: 1; }
        .comment-author {
            font-weight: 700;
            color: var(--text-dark);
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .author-badge {
            background: var(--teal-primary);
            color: white;
            font-size: 0.65rem;
            padding: 2px 8px;
            border-radius: 10px;
            text-transform: uppercase;
            font-weight: 800;
        }
        .comment-text {
            color: #475569;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )