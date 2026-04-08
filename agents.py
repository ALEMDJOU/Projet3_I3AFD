"""
agents.py — Définition de l'état, des nœuds agents et construction du graphe LangGraph.
"""
import re
import random
import operator
import requests
import streamlit as st
from typing import Annotated, List, TypedDict

from googleapiclient.discovery import build
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ──────────────────────────────────────────────
# État partagé du pipeline
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    video_id:         str
    video_info:       dict
    raw_comments:     List[str]
    filtered_comments: List[str]
    analyses:         Annotated[list, operator.add]
    final_score:      float
    summary:          str
    reflection_logs:  Annotated[list, operator.add]
    sentiment_scores: List[float]


# ──────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────

def extract_video_id(url_or_id: str) -> str:
    """Extrait l'ID YouTube depuis une URL ou retourne la chaîne brute."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url_or_id)
    return match.group(1) if match else url_or_id.strip()


def call_gemini_api(prompt: str, gemini_key: str) -> str:
    """
    Appelle l'API Gemini (générative).
    La clé est passée en paramètre — aucun couplage avec st.session_state.
    """
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}"
    try:
        r_list = requests.get(list_url, timeout=10)
        if r_list.status_code != 200:
            return f"ERREUR LISTAGE ({r_list.status_code}): {r_list.text}"

        models = r_list.json().get("models", [])
        available = [
            m["name"] for m in models
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        if not available:
            return "ERREUR : aucun modèle de génération disponible sur cette clé."

        gen_url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"{available[0]}:generateContent?key={gemini_key}"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        r_gen = requests.post(gen_url, headers={"Content-Type": "application/json"},
                              json=payload, timeout=20)
        if r_gen.status_code == 200:
            return r_gen.json()["candidates"][0]["content"]["parts"][0]["text"]
        return f"ERREUR GÉNÉRATION ({r_gen.status_code}): {r_gen.text}"

    except Exception as exc:
        return f"ERREUR RÉSEAU: {exc}"


# ──────────────────────────────────────────────
# Nœuds agents
# ──────────────────────────────────────────────

def make_fetch_node(youtube_key: str):
    """Factory : retourne un nœud fetcher configuré avec la clé YouTube."""
    def fetch_comments_node(state: AgentState):
        try:
            youtube = build("youtube", "v3", developerKey=youtube_key)
            v_res = youtube.videos().list(part="snippet", id=state["video_id"]).execute()
            info = {"title": "Vidéo Inconnue", "author": "Inconnu"}
            if v_res["items"]:
                info["title"]  = v_res["items"][0]["snippet"]["title"]
                info["author"] = v_res["items"][0]["snippet"]["channelTitle"]

            response = youtube.commentThreads().list(
                part="snippet", videoId=state["video_id"],
                maxResults=100, order="relevance"
            ).execute()
            all_comments = [
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for item in response["items"]
            ]
            # Tirage aléatoire reproductible : 30 sur 100
            random.seed(42)
            comments = random.sample(all_comments, min(30, len(all_comments)))
            log = f"Fetcher : {len(all_comments)} commentaires récupérés → {len(comments)} tirés aléatoirement (seed=42) pour « {info['title']} »."
            return {"raw_comments": comments, "video_info": info, "reflection_logs": [log]}

        except Exception as exc:
            return {
                "raw_comments": [],
                "video_info": {"title": "Erreur", "author": "API Error"},
                "reflection_logs": [f"Fetcher : Erreur API YouTube — {exc}"],
            }

    return fetch_comments_node


def filter_agent(state: AgentState):
    raw   = state.get("raw_comments", [])
    clean = [re.sub(r"<[^>]+>", " ", c) for c in raw if len(c) > 30]
    # Filtre anti-spam basique (URLs)
    clean = [c for c in clean if not re.search(r"http|www\.|\.com|\.net", c)]
    log   = f"Filter : {len(clean)} commentaires conservés sur {len(raw)} (spam + bruit retirés)."
    return {"filtered_comments": clean, "reflection_logs": [log]}


def make_analyst_node(gemini_key: str):
    """Factory : retourne un nœud analyste configuré avec la clé Gemini."""
    def analyst_agent(state: AgentState):
        comments = state["filtered_comments"]
        if not comments:
            return {"analyses": ["Aucun commentaire pertinent."],
                    "sentiment_scores": [], "reflection_logs": ["Analyst : aucun commentaire."]}

        comments_block = "\n---\n".join(comments)
        # Scores de sentiment par heuristique légère (évite un appel API par commentaire)
        scores = [
            0.85 if any(w in c.lower() for w in ["good", "great", "merci", "super", "excellent"])
            else 0.35 if any(w in c.lower() for w in ["bad", "nul", "mauvais", "terrible"])
            else 0.55
            for c in comments
        ]
        prompt = (
            "Analyse ces commentaires YouTube. Identifie le sentiment général "
            "(positif, négatif, neutre), la qualité du discours et si le contenu "
            "de la vidéo semble instructif. Sois précis et structuré.\n\n"
            f"{comments_block}"
        )
        result = call_gemini_api(prompt, gemini_key)
        log    = f"Analyst : {len(comments)} commentaires traités. Extrait : {result[:60]}…"
        return {"analyses": [result], "sentiment_scores": scores, "reflection_logs": [log]}

    return analyst_agent


def make_synthesizer_node(gemini_key: str):
    """Factory : retourne un nœud synthesizer configuré avec la clé Gemini."""
    def synthesis_agent(state: AgentState):
        full_analysis = "\n".join(state["analyses"])
        prompt = (
            f"Basé sur cette analyse :\n{full_analysis}\n\n"
            "Donne une note sur 10 et un résumé court (max 3 phrases) de la qualité "
            "globale de la vidéo.\nFormat exact : SCORE: X/10 | RÉSUMÉ: Texte"
        )
        result = call_gemini_api(prompt, gemini_key)
        try:
            score = float(result.split("|")[0].split(":")[1].split("/")[0].strip())
        except Exception:
            score = 0.0
        log = f"Synthesizer : Score calculé = {score}/10."
        return {"final_score": score, "summary": result, "reflection_logs": [log]}

    return synthesis_agent


# ──────────────────────────────────────────────
# Construction du graphe
# ──────────────────────────────────────────────

def build_graph(youtube_key: str, gemini_key: str):
    """
    Assemble et compile le graphe LangGraph.
    Retourne l'application compilée, prête à être streamée.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("fetcher",     make_fetch_node(youtube_key))
    workflow.add_node("filter",      filter_agent)
    workflow.add_node("analyst",     make_analyst_node(gemini_key))
    workflow.add_node("synthesizer", make_synthesizer_node(gemini_key))

    workflow.set_entry_point("fetcher")
    workflow.add_edge("fetcher",     "filter")
    workflow.add_edge("filter",      "analyst")
    workflow.add_edge("analyst",     "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile(checkpointer=MemorySaver())