"""
agents.py — Définition de l'état, des nœuds agents et construction du graphe LangGraph.
           Version Master : recherche YouTube, métadonnées, filtrage avancé,
           analyse 7 dimensions, fallback Gemini → Qwen → DeepSeek.
           Récupère jusqu'à 300 commentaires via 3 appels API paginés.
           Ajout des logs de réflexion pour chaque nœud.
"""
import re
import time
import html
import operator
import requests
from typing import Annotated, List, TypedDict, Tuple, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ══════════════════════════════════════════════════════════════════════════════
#  ÉTAT PARTAGÉ
# ══════════════════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    # Entrée utilisateur
    search_query: str
    video_id: Optional[str]

    # Métadonnées vidéo
    video_title: str
    video_channel: str
    video_views: str
    video_likes: str
    video_description: str

    # Commentaires
    raw_comments: List[str]
    filtered_comments: List[str]
    spam_count: int

    # Analyses
    analyses: Annotated[list, operator.add]

    # Résultat final
    final_score: float
    confidence: str
    recommendation: str
    summary: str
    pipeline_error: Optional[str]

    # Métriques additionnelles pour l'UI
    sentiment_scores: List[float]
    polarity: float
    instructive_score: float

    # Logs (ajouté)
    reflection_logs: Annotated[list, operator.add]


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    youtube_api_key: str = None
    gemini_api_key: str = None
    hf_api_key: str = None

    max_comments: int = 300
    max_pages: int = 3
    youtube_order: str = "relevance"

    min_comment_length: int = 20
    max_comment_length: int = 400
    max_comments_to_llm: int = 60

    hf_timeout: int = 60
    hf_retries: int = 3

    yt_search_url: str = "https://www.googleapis.com/youtube/v3/search"
    yt_videos_url: str = "https://www.googleapis.com/youtube/v3/videos"
    yt_comments_url: str = "https://www.googleapis.com/youtube/v3/commentThreads"
    hf_chat_url: str = "https://router.huggingface.co/v1/chat/completions"

cfg = Config()


def set_api_keys(keys: dict):
    cfg.youtube_api_key = keys["youtube"]
    cfg.gemini_api_key = keys["gemini"]
    cfg.hf_api_key = keys["huggingface"]


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════
def extract_video_id(url_or_id: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url_or_id)
    return match.group(1) if match else url_or_id.strip()


def clean_html_comment(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", "[URL]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_spam(comment: str) -> bool:
    lower = comment.lower()
    spam_patterns = [
        r"sub.*sub", r"check.*my.*channel", r"follow.*back",
        r"\b(like|subscribe|notification)\b.{0,20}\b(please|pls|plz)\b",
        r"free.*(?:robux|gems|coins|gift)", r"click.*link.*bio",
        r"(?:first|premier|primero)\s*[!🔥👀]+$",
    ]
    if any(re.search(p, lower) for p in spam_patterns):
        return True
    if re.search(r"(.)\1{5,}", comment):
        return True
    emoji_count = len(re.findall(r"[^\w\s,.\-!?;:'\"]", comment))
    if len(comment) > 0 and emoji_count / len(comment) > 0.4:
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  APPELS LLM AVEC FALLBACK (Gemini → Qwen → DeepSeek)
# ══════════════════════════════════════════════════════════════════════════════
def call_gemini_api(prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={cfg.gemini_api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        elif r.status_code in (429, 500, 503):
            return "FALLBACK"
        return f"ERREUR Gemini {r.status_code}"
    except:
        return "FALLBACK"


def call_huggingface_model(prompt: str, model_id: str, system: Optional[str] = None) -> str:
    headers = {"Authorization": f"Bearer {cfg.hf_api_key}", "Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.3,
    }
    for attempt in range(cfg.hf_retries):
        try:
            r = requests.post(cfg.hf_chat_url, headers=headers, json=payload, timeout=cfg.hf_timeout)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            if r.status_code == 429:
                time.sleep(2 ** attempt * 2)
                continue
        except:
            time.sleep(2)
    return "FALLBACK"


def call_llm_with_fallback(prompt: str, system: Optional[str] = None) -> Tuple[str, str]:
    print("🔁 Tentative avec Gemini...")
    res = call_gemini_api(prompt)
    if "FALLBACK" not in res and "ERREUR" not in res:
        print("✅ Gemini OK")
        return res, "Gemini"
    print("⚠️ Gemini échoue, passage à Qwen...")
    res = call_huggingface_model(prompt, "Qwen/Qwen2.5-7B-Instruct", system)
    if "FALLBACK" not in res and "ERREUR" not in res:
        print("✅ Qwen OK")
        return res, "Qwen 2.5"
    print("⚠️ Qwen échoue, passage à DeepSeek...")
    res = call_huggingface_model(prompt, "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", system)
    if "FALLBACK" not in res and "ERREUR" not in res:
        print("✅ DeepSeek OK")
        return res, "DeepSeek"
    return "ERREUR: Aucun modèle disponible", "Aucun"

# ══════════════════════════════════════════════════════════════════════════════
#  NŒUDS LANGGRAPH (avec logs)
# ══════════════════════════════════════════════════════════════════════════════

def node_search(state: AgentState) -> dict:
    if state.get("video_id"):
        return {"reflection_logs": ["Search : vidéo déjà fournie, recherche ignorée."]}
    query = state.get("search_query", "").strip()
    if not query:
        return {"pipeline_error": "Aucune requête ni video_id fourni.", "reflection_logs": ["Search : erreur - pas de requête."]}
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": 1,
        "relevanceLanguage": "fr",
        "key": cfg.youtube_api_key,
    }
    try:
        r = requests.get(cfg.yt_search_url, params=params, timeout=10)
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return {"pipeline_error": f"Aucune vidéo trouvée pour '{query}'", "reflection_logs": ["Search : aucun résultat."]}
        video_id = items[0]["id"]["videoId"]
        log = f"Search : vidéo trouvée → {video_id}"
        return {"video_id": video_id, "reflection_logs": [log]}
    except Exception as e:
        return {"pipeline_error": f"Erreur recherche YouTube: {e}", "reflection_logs": [f"Search : erreur - {e}"]}


def node_metadata(state: AgentState) -> dict:
    video_id = state.get("video_id")
    if not video_id:
        return {"pipeline_error": "Pas de video_id", "reflection_logs": ["Metadata : pas de video_id"]}
    params = {
        "part": "snippet,statistics",
        "id": video_id,
        "key": cfg.youtube_api_key,
    }
    try:
        r = requests.get(cfg.yt_videos_url, params=params, timeout=10)
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return {"video_title": "Inconnue", "video_channel": "Inconnue", "video_views": "N/A", "video_likes": "N/A", "video_description": "", "reflection_logs": ["Metadata : aucune donnée"]}
        s = items[0]["snippet"]
        stats = items[0].get("statistics", {})
        title = s.get("title", "N/A")
        log = f"Metadata : {title} | {stats.get('viewCount',0)} vues | {stats.get('likeCount',0)} likes"
        return {
            "video_title": title,
            "video_channel": s.get("channelTitle", "N/A"),
            "video_views": f"{int(stats.get('viewCount',0)):,}",
            "video_likes": f"{int(stats.get('likeCount',0)):,}",
            "video_description": s.get("description", "")[:2000],
            "reflection_logs": [log],
        }
    except Exception as e:
        return {"pipeline_error": f"Erreur métadonnées: {e}", "reflection_logs": [f"Metadata : erreur - {e}"]}


def node_fetch_comments(state: AgentState) -> dict:
    video_id = state.get("video_id")
    if not video_id:
        return {"raw_comments": [], "spam_count": 0, "reflection_logs": ["Fetch : pas de video_id"]}
    all_comments = []
    next_token = None
    pages_done = 0
    while pages_done < cfg.max_pages and len(all_comments) < cfg.max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "order": cfg.youtube_order,
            "key": cfg.youtube_api_key,
        }
        if next_token:
            params["pageToken"] = next_token
        try:
            r = requests.get(cfg.yt_comments_url, params=params, timeout=15)
            if r.status_code == 403:
                break
            r.raise_for_status()
            data = r.json()
            for item in data.get("items", []):
                text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                all_comments.append(text)
            next_token = data.get("nextPageToken")
            pages_done += 1
            if not next_token:
                break
        except:
            break
    log = f"Fetch : {len(all_comments)} commentaires récupérés ({pages_done} pages)"
    return {"raw_comments": all_comments, "reflection_logs": [log]}


def node_filter(state: AgentState) -> dict:
    raw = state.get("raw_comments", [])
    spam_count = 0
    seen = set()
    filtered = []
    for c in raw:
        clean = clean_html_comment(c)
        if len(clean) < cfg.min_comment_length:
            spam_count += 1
            continue
        if is_spam(clean):
            spam_count += 1
            continue
        fp = clean[:80].lower()
        if fp in seen:
            spam_count += 1
            continue
        seen.add(fp)
        filtered.append(clean[:cfg.max_comment_length])
    filtered_for_llm = filtered[:cfg.max_comments_to_llm]
    log = f"Filter : {len(raw)} → {len(filtered_for_llm)} commentaires conservés (spam: {spam_count})"
    return {"filtered_comments": filtered_for_llm, "spam_count": spam_count, "reflection_logs": [log]}


def node_analyst(state: AgentState) -> dict:
    comments = state.get("filtered_comments", [])
    if not comments:
        return {"analyses": ["Aucun commentaire à analyser."], "sentiment_scores": [], "polarity": 0.0, "instructive_score": 0.0, "reflection_logs": ["Analyst : aucun commentaire"]}
    
    # Heuristique pour timeline
    sentiment_scores = []
    for c in comments:
        c_low = c.lower()
        if any(w in c_low for w in ["good", "great", "merci", "super", "excellent", "awesome"]):
            sentiment_scores.append(0.85)
        elif any(w in c_low for w in ["bad", "nul", "mauvais", "terrible", "horrible"]):
            sentiment_scores.append(0.35)
        else:
            sentiment_scores.append(0.55)
    polarity = (sum(sentiment_scores) / len(sentiment_scores) - 0.55) / 0.3
    polarity = max(-1.0, min(1.0, polarity))
    instructive_score = 0.5

    comments_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(comments[:50]))
    query = state.get("search_query", "le sujet de la vidéo")
    title = state.get("video_title", "N/A")
    channel = state.get("video_channel", "N/A")
    views = state.get("video_views", "N/A")
    likes = state.get("video_likes", "N/A")
    desc = state.get("video_description", "")[:1000]

    system = "Tu es un expert en analyse de contenu vidéo. Réponds de manière structurée et précise."
    user_prompt = f"""
## CONTEXTE VIDÉO
- Requête : "{query}"
- Titre : {title}
- Chaîne : {channel}
- Vues : {views} · Likes : {likes}
- Description : {desc or "(non disponible)"}

## COMMENTAIRES ({len(comments)} analysés, extrait)
{comments_block}

## TÂCHE (réponds section par section)
1. SIMILARITÉ REQUÊTE/CONTENU [0-10]
2. ANALYSE SENTIMENTALE (répartition % positifs/négatifs/neutres)
3. VALIDATION PÉDAGOGIQUE (exemples)
4. THÈMES RÉCURRENTS (top 3)
5. SCORE D'UTILITÉ PERÇUE [0-10]
6. CATÉGORISATION (type, audience)
7. SIGNAUX D'ALERTE
8. SCORE INSTRUCTIF [0-1]
9. POLARITÉ GLOBALE [-1 à 1]
"""
    response, model_used = call_llm_with_fallback(user_prompt, system)
    try:
        match_instr = re.search(r"(?:score instructif|instructif)[\s:]*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
        if match_instr:
            instructive_score = float(match_instr.group(1))
        match_pol = re.search(r"(?:polarité|polarite)[\s:]*([-]?[0-9]*\.?[0-9]+)", response, re.IGNORECASE)
        if match_pol:
            polarity = float(match_pol.group(1))
            polarity = max(-1.0, min(1.0, polarity))
    except:
        pass
    analysis_text = f"**Modèle utilisé : {model_used}**\n\n{response}"
    log = f"Analyst : {len(comments)} commentaires analysés avec {model_used}"
    return {
        "analyses": [analysis_text],
        "sentiment_scores": sentiment_scores,
        "polarity": polarity,
        "instructive_score": instructive_score,
        "reflection_logs": [log],
    }


def node_synthesizer(state: AgentState) -> dict:
    full_analysis = "\n\n".join(state.get("analyses", []))
    query = state.get("search_query", "le sujet")
    title = state.get("video_title", "N/A")
    nb_comments = len(state.get("filtered_comments", []))
    spam_count = state.get("spam_count", 0)

    system = "Tu es un synthétiseur. Respecte le format exact."
    user_prompt = f"""
## DONNÉES
- Requête : "{query}"
- Vidéo : {title}
- Commentaires analysés : {nb_comments} (spam filtré : {spam_count})

## ANALYSE DÉTAILLÉE
{full_analysis}

## TÂCHE : SYNTHÈSE FINALE
Barème : 9-10 exceptionnel, 7-8 très bon, 5-6 moyen, 3-4 problèmes, 0-2 à éviter.
Confiance : Haute (>50 comm. consensus clair), Moyenne (20-50 ou avis divisés), Faible (<20).

Réponds EXACTEMENT au format :
SCORE: X.X/10
CONFIANCE: haute|moyenne|faible
RECOMMANDATION: À regarder|Selon vos intérêts|À éviter
RÉSUMÉ: [3 phrases max]
"""
    response, model_used = call_llm_with_fallback(user_prompt, system)
    score = 0.0
    confidence = "faible"
    recommendation = "Selon vos intérêts"
    summary = response
    m_score = re.search(r"SCORE\s*:\s*([\d\.]+)/10", response, re.IGNORECASE)
    if m_score:
        score = float(m_score.group(1))
    m_conf = re.search(r"CONFIANCE\s*:\s*(haute|moyenne|faible)", response, re.IGNORECASE)
    if m_conf:
        confidence = m_conf.group(1).lower()
    if "À regarder" in response:
        recommendation = "À regarder"
    elif "À éviter" in response:
        recommendation = "À éviter"
    m_sum = re.search(r"RÉSUMÉ\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if m_sum:
        summary = m_sum.group(1).strip()
    log = f"Synthesizer : score {score}/10, confiance {confidence}, recommandation {recommendation}"
    return {
        "final_score": score,
        "confidence": confidence,
        "recommendation": recommendation,
        "summary": summary,
        "reflection_logs": [log],
    }


def should_continue(state: AgentState) -> str:
    if state.get("pipeline_error"):
        return "abort"
    if not state.get("video_id"):
        return "abort"
    return "continue"


def node_abort(state: AgentState) -> dict:
    error = state.get("pipeline_error", "Erreur inconnue")
    return {
        "final_score": 0.0,
        "confidence": "faible",
        "recommendation": "Données insuffisantes",
        "summary": f"Analyse impossible : {error}",
        "pipeline_error": error,
        "reflection_logs": [f"Abort : {error}"],
    }


def build_graph(keys: dict):
    set_api_keys(keys)
    workflow = StateGraph(AgentState)
    workflow.add_node("search", node_search)
    workflow.add_node("metadata", node_metadata)
    workflow.add_node("fetch", node_fetch_comments)
    workflow.add_node("filter", node_filter)
    workflow.add_node("analyst", node_analyst)
    workflow.add_node("synthesizer", node_synthesizer)
    workflow.add_node("abort", node_abort)

    workflow.set_entry_point("search")
    workflow.add_conditional_edges("search", should_continue, {"continue": "metadata", "abort": "abort"})
    workflow.add_edge("metadata", "fetch")
    workflow.add_edge("fetch", "filter")
    workflow.add_edge("filter", "analyst")
    workflow.add_edge("analyst", "synthesizer")
    workflow.add_edge("synthesizer", END)
    workflow.add_edge("abort", END)

    return workflow.compile(checkpointer=MemorySaver())