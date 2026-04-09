"""
app.py — Point d'entrée AVA Pro.
         Orchestre config, graphe, UI et métriques.
"""
import os
import streamlit as st
import plotly.graph_objects as go

from config  import setup_page, get_api_keys
from agents  import build_graph, extract_video_id
from metrics import render_ablation

# ──────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────

setup_page()
keys = get_api_keys()

UI_STRINGS = {
    "fr": {
        "dashboard_title": "Tableau de Bord AVA PRO",
        "dashboard_subtitle": "Analyseur de vidéos YouTube basé sur les commentaires · Swarm multi-agents LangGraph",
        "analyze_title": "Analyser une vidéo",
        "url_placeholder": "URL ou ID YouTube",
        "btn_run": "🚀 LANCER L'ANALYSE",
        "btn_stop": "🛑 ARRÊTER L'ANALYSE",
        "exec_trace": " Suivi de l'Exécution",
        "logs": "LOGS",
        "success": "✅ Analyse terminée.",
        "results_title": " Résultats de l'Analyse",
        "gauge_title": "QUALITÉ GLOBALE",
        "timeline_title": "ÉVOLUTION DU SENTIMENT",
        "meta_source": "SOURCE",
        "analyst_name": "AVA Analyst",
        "analyst_badge": "Expert Swarm",
        "tabs": ["Bruts", "Filtrés", "Analyse"],
        "step_names": {
            "search": "Recherche",
            "fetch": "Fetcher",
            "filter": "Filter",
            "analyst": "Analyst",
            "synthesizer": "Synthesizer"
        },
        "team_title": "L'Équipe de Développement AVA PRO",
        
        "welcome_message": "Bienvenue sur AVA Pro ! Votre outil d'analyse vidéo YouTube basé sur l'intelligence artificielle. Lancez une analyse pour découvrir les insights cachés dans les commentaires.",
    },
    
    "en": {
        "dashboard_title": "AVA PRO Dashboard",
        "dashboard_subtitle": "YouTube video analyzer based on comments · Multi-agent swarm semantic analysis",
        "analyze_title": "Analyze a Video",
        "url_placeholder": "YouTube URL or ID",
        "btn_run": "🚀 START ANALYSIS",
        "btn_stop": "🛑 STOP ANALYSIS",
        "exec_trace": " Execution Tracking",
        "logs": "LOGS",
        "success": "✅ Analysis finished.",
        "results_title": " Analysis Results",
        "gauge_title": "GLOBAL QUALITY",
        "timeline_title": "SENTIMENT EVOLUTION",
        "meta_source": "SOURCE",
        "analyst_name": "AVA Analyst",
        "analyst_badge": "Swarm Expert",
        "tabs": ["Raw", "Filtered", "Analysis"],
        "step_names": {
            "search": "Search",
            "fetch": "Fetcher",
            "filter": "Filter",
            "analyst": "Analyst",
            "synthesizer": "Synthesizer"
        },
        "team_title": "AVA PRO Development Team",
        
        "welcome_message": "Welcome to AVA Pro! Your AI-powered YouTube video analysis tool. Start an analysis to uncover hidden insights in comments.",
    }
}

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    if os.path.exists("logoAVA.png"):
        st.image("logoAVA.png", width=250)
    else:
        st.markdown("<h1 style='text-align:center;color:#00D4AC;'>AVA PRO</h1>",
                    unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;color:#64748B;font-size:0.9rem;"
                "margin-top:-10px;'>Agentic Video Analysis v3.0</p>",
                unsafe_allow_html=True)
    st.markdown("---")
    lang = st.selectbox(
        "🌐 Language / Langue",
        options=["fr", "en"],
        format_func=lambda x: "Français" if x == "fr" else "English"
    )
    t = UI_STRINGS[lang]
    st.markdown("---")
    st.markdown("<h3 style='font-size:1.1rem;'>"
                "<i class='fa-solid fa-layer-group'></i> ARCHITECTURE</h3>",
                unsafe_allow_html=True)

    # Diagramme SVG circular
    svg = """
    <svg width="260" height="260" viewBox="0 0 260 260" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="tg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%"   style="stop-color:#00D4AC"/>
          <stop offset="100%" style="stop-color:#018EA9"/>
        </linearGradient>
        <marker id="arr" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
          <polygon points="0 0,10 3.5,0 7" fill="#018EA9"/>
        </marker>
      </defs>
      <circle cx="130" cy="130" r="75" fill="none" stroke="#E2E8F0"
              stroke-width="1.5" stroke-dasharray="6,4"/>
      <g><circle cx="130" cy="55"  r="24" fill="white" stroke="url(#tg)" stroke-width="2.5"/>
         <text x="130" y="61"  font-size="18" text-anchor="middle" fill="#018EA9"
               font-family="'Font Awesome 6 Free'" font-weight="900">&#xf019;</text>
         <text x="130" y="20"  font-size="13" font-weight="800" text-anchor="middle"
               fill="#1E293B" font-family="Inter,sans-serif">Fetcher</text></g>
      <g><circle cx="205" cy="130" r="24" fill="white" stroke="url(#tg)" stroke-width="2.5"/>
         <text x="205" y="136" font-size="18" text-anchor="middle" fill="#018EA9"
               font-family="'Font Awesome 6 Free'" font-weight="900">&#xf0b0;</text>
         <text x="255" y="135" font-size="13" font-weight="800" text-anchor="middle"
               fill="#1E293B" font-family="Inter,sans-serif">Filter</text></g>
      <g><circle cx="130" cy="205" r="24" fill="white" stroke="url(#tg)" stroke-width="2.5"/>
         <text x="130" y="211" font-size="18" text-anchor="middle" fill="#018EA9"
               font-family="'Font Awesome 6 Free'" font-weight="900">&#xf5dc;</text>
         <text x="130" y="247" font-size="13" font-weight="800" text-anchor="middle"
               fill="#1E293B" font-family="Inter,sans-serif">Analyst</text></g>
      <g><circle cx="55"  cy="130" r="24" fill="white" stroke="url(#tg)" stroke-width="2.5"/>
         <text x="55"  y="136" font-size="18" text-anchor="middle" fill="#018EA9"
               font-family="'Font Awesome 6 Free'" font-weight="900">&#xf303;</text>
         <text x="5"   y="135" font-size="13" font-weight="800" text-anchor="middle"
               fill="#1E293B" font-family="Inter,sans-serif">Synth</text></g>
      <path d="M148 68 L187 107" stroke="#018EA9" stroke-width="2.5" fill="none" marker-end="url(#arr)"/>
      <path d="M187 153 L148 192" stroke="#018EA9" stroke-width="2.5" fill="none" marker-end="url(#arr)"/>
      <path d="M112 192 L73 153"  stroke="#018EA9" stroke-width="2.5" fill="none" marker-end="url(#arr)"/>
      <path d="M73 107 L112 68"   stroke="#018EA9" stroke-width="2.5" fill="none" marker-end="url(#arr)"/>
    </svg>
    """
    st.markdown(svg.replace("<svg", "<svg style='max-width: 100%; height: auto;'"), unsafe_allow_html=True) # Added style to SVG

# ──────────────────────────────────────────────
# En-tête principal
# ──────────────────────────────────────────────

st.markdown(
    '<div class="main-header">'
    f'<h1 style="margin:0;font-size:2.5rem;font-weight:800;">{t["dashboard_title"]}</h1>'
    '<p style="margin:5px 0 0 0;opacity:0.9;font-size:1.1rem;">'
    f'{t["dashboard_subtitle"]}'
    '</p></div>',
    unsafe_allow_html=True,
)
st.info(t["welcome_message"])

# ──────────────────────────────────────────────
# Saisie vidéo
# ──────────────────────────────────────────────

st.markdown(
    f"<div class='section-title'><i class='fa-solid fa-link'></i> {t['analyze_title']}</div>",
    unsafe_allow_html=True,
)
col_input, col_btn = st.columns([4, 1.2])
with col_input:
    video_input = st.text_input(
        t["url_placeholder"],
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
with col_btn:
    run_button = st.button(t["btn_run"], type="primary")

# ──────────────────────────────────────────────
# Exécution du pipeline
# ──────────────────────────────────────────────

if run_button:
    if not keys["youtube"] or not keys["gemini"]:
        st.error("Configuration incomplète : clés API manquantes dans le fichier .env.")
    elif not video_input:
        st.warning("Veuillez saisir une URL ou un ID YouTube.")
    else:
        video_id  = extract_video_id(video_input)
        app_graph = build_graph(keys)
        config    = {"configurable": {"thread_id": "ava_pro_session"}}

        st.markdown(
            "<div class='section-title'><i class='fa-solid fa-microchip'></i>"
            f"{t['exec_trace']}</div>",
            unsafe_allow_html=True,
        )
        col_exec, col_logs = st.columns([2, 1])
        with col_exec:
            tracker_box  = st.empty()
            progress_bar = st.progress(0)
            if st.button(t["btn_stop"], type="secondary", use_container_width=True):
                st.rerun()
        with col_logs:
            st.markdown("<h4 style='font-size:0.9rem;color:#64748B;'>LOGS</h4>",
                        unsafe_allow_html=True)
            log_container = st.empty()

        steps_display = [
            ("search",      t["step_names"]["search"],      "fa-magnifying-glass"),
            ("fetch",       t["step_names"]["fetch"],       "fa-download"),
            ("filter",      t["step_names"]["filter"],      "fa-filter"),
            ("analyst",     t["step_names"]["analyst"],     "fa-brain"),
            ("synthesizer", t["step_names"]["synthesizer"], "fa-wand-magic-sparkles"),
        ]
        completed = []

        loading_msg = "AVA Pro : Analyseur de vidéos YouTube basé sur les commentaires. Swarm multi-agents en action..." if lang == "fr" else "AVA Pro: YouTube video analyzer based on comments. Multi-agent swarm in action..."
        with st.spinner(loading_msg):
            for i, output in enumerate(app_graph.stream({"video_id": video_id, "language": lang}, config)):
                active_node = list(output.keys())[0]
                completed.append(active_node)

                state = app_graph.get_state(config).values
                logs  = state.get("reflection_logs", [])

                # Logs
                log_html = ("<div class='ava-log-container' style='height:150px;overflow-y:auto;font-size:0.8rem;" # Added class
                            "color:#334155;border:1px solid #E2E8F0;padding:10px;"
                            "border-radius:10px;background:#F8FAFF;'>")
                for log in reversed(logs):
                    log_html += f"<div style='margin-bottom:5px;'>• {log}</div>"
                log_html += "</div>"
                log_container.markdown(log_html, unsafe_allow_html=True)

                # Tracker
                tracker_html = "<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:10px;'>"
                for sid, sname, icon in steps_display:
                    done   = sid in completed
                    active = active_node == sid
                    color  = "#10B981" if done else "#00D4AC" if active else "#94A3B8"
                    ic     = "fa-circle-check" if done else icon
                    css    = "step-active" if active else ""
                    tracker_html += (
                        f'<div class="step-item {css}">'
                        f'<i class="fa-solid {ic}" style="color:{color};"></i> {sname}</div>'
                    )
                tracker_html += "</div>"
                tracker_box.markdown(tracker_html, unsafe_allow_html=True)
                progress_bar.progress(min((i + 1) / len(steps_display), 1.0))

        st.success(t["success"])

        # ──────────────────────────────────────────────
        # Affichage des résultats
        # ──────────────────────────────────────────────

        final_state = app_graph.get_state(config).values
        v_title = final_state.get("video_title", "N/A")
        v_author = final_state.get("video_channel", "N/A")
        score = final_state.get("final_score", 0)

        st.markdown("---")
        st.markdown(
            "<div class='section-title'><i class='fa-solid fa-chart-pie'></i>"
            f"{t['results_title']}</div>",
            unsafe_allow_html=True,
        )

        col_gauge, col_trend, col_info = st.columns([1.5, 2, 2.5])

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": t["gauge_title"], "font": {"size": 16, "color": "#018EA9"}},
                gauge={
                    "axis": {"range": [None, 10]},
                    "bar":  {"color": "#00D4AC"},
                    "steps": [
                        {"range": [0, 5], "color": "#FEE2E2"},
                        {"range": [5, 8], "color": "#FEF3C7"},
                        {"range": [8, 10], "color": "#D1FAE5"},
                    ],
                    "threshold": {"line": {"color": "#10B981", "width": 4},
                                  "thickness": 0.75, "value": 9.0},
                },
            ))
            fig_gauge.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=20),
                                    font={"family": "Inter"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_trend:
            scores = final_state.get("sentiment_scores", [])
            if scores:
                fig_timeline = go.Figure()
                fig_timeline.add_trace(go.Scatter(
                    y=scores, mode="lines+markers",
                    line=dict(color="#00D4AC", width=3),
                    marker=dict(size=8, color="#018EA9"),
                    fill="tozeroy", fillcolor="rgba(0,212,172,0.1)",
                ))
                fig_timeline.update_layout(
                    title={"text": t["timeline_title"],
                           "font": {"size": 14, "color": "#018EA9"}},
                    height=230, margin=dict(l=0, r=0, t=50, b=0),
                    xaxis={"showgrid": False, "showticklabels": False},
                    yaxis={"range": [0, 1], "gridcolor": "#F1F5F9"},
                    plot_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"},
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

        with col_info:
            summary_text = (
                final_state.get("summary", "")
                .split("|")[-1]
                .replace("RÉSUMÉ:", "").replace("SUMMARY:", "")
                .strip()
            )
            # Carte Métadonnées
            st.markdown(
                f'<div class="info-card">'
                f'<h4 style="color:#64748B;margin:0;font-size:0.75rem;letter-spacing:1px;">'
                f'{t["meta_source"]}</h4>'
                f'<p style="font-weight:700;color:#1E293B;font-size:1.05rem;margin:5px 0;">'
                f'{v_title}</p>'
                f'<p style="color:#00D4AC;font-size:0.9rem;margin:0;">'
                f'<i class="fa-solid fa-circle-user"></i> {v_author}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Bulle de commentaire de l'Analyste
            st.markdown(
                f'<div class="comment-box">'
                f'  <div class="analyst-avatar"><i class="fa-solid fa-brain"></i></div>'
                f'  <div class="comment-content">'
                f'    <div class="comment-author">{t["analyst_name"]} '
                f'<span class="author-badge">{t["analyst_badge"]}</span></div>'
                f'    <div class="comment-text">"{summary_text}"</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Tabs commentaires
        tab1, tab2, tab3 = st.tabs(t["tabs"])
        with tab1:
            for c in final_state.get("raw_comments", [])[:10]:
                st.markdown(
                    f"<div style='font-size:0.9rem;padding:8px;"
                    f"border-bottom:1px solid #F1F5F9;'>{c[:200]}…</div>",
                    unsafe_allow_html=True,
                )
        with tab2:
            for i, c in enumerate(final_state.get("filtered_comments", [])[:10], 1):
                icon = ("<i class='fa-solid fa-face-smile' style='color:#10B981;'></i>"
                        if i % 2 == 0 else
                        "<i class='fa-solid fa-face-meh' style='color:#F87171;'></i>")
                st.markdown(
                    f"<div style='font-size:0.9rem;padding:10px;background:white;"
                    f"border-radius:8px;margin-bottom:5px;'>{icon} {c[:200]}…</div>",
                    unsafe_allow_html=True,
                )
        with tab3:
            st.info(final_state.get("analyses", ["N/A"])[0])

        # ──────────────────────────────────────────────
        # Ablation study (métriques HuggingFace)
        # ──────────────────────────────────────────────
        render_ablation(final_state, None, keys["gemini"], lang=lang)
# ──────────────────────────────────────────────
# Section Équipe de Développement (Footer)
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div class='section-title'><i class='fa-solid fa-users'></i> {t['team_title']}</div>",
    unsafe_allow_html=True
)

# Chemins des images (à ajuster selon votre structure)
image1 = "images/proje3.jpg"      # 3840x2160
image2 = "images/projet.jpg"   
image3 = "images/i3afd.jpg"     # 2160x3828

# CSS pour un affichage élégant
st.markdown("""
<style>
    /* Création d'un bloc unique pour l'équipe */
    [data-testid="stHorizontalBlock"] {
        gap: 0px !important;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #E2E8F0;
        background: white;
    }
    /* Ajustement des images pour qu'elles remplissent le bloc sans vide */
    [data-testid="column"] img {
        height: 280px !important; /* Hauteur fixe par défaut */
        object-fit: cover !important;
        width: 100% !important;
        border-radius: 0px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="column"] img:hover {
        transform: scale(1.05);
        z-index: 10;
    }

    /* Media queries pour la responsivité */
    @media (max-width: 768px) { /* Pour les tablettes et plus petits */
        [data-testid="column"] img {
            height: 200px !important; /* Réduire la hauteur */
        }
    }
    @media (max-width: 480px) { /* Pour les téléphones mobiles */
        [data-testid="stHorizontalBlock"] {
            gap: 0px !important; /* Pas d'espace entre les images sur mobile */
        }
        [data-testid="column"] img {
            height: 220px !important; /* Plus de hauteur pour un rendu "carte" sur mobile */
        }
    }

    /* Responsive adjustments for specific elements in app.py */
    @media (max-width: 768px) { /* Tablets and smaller */
        .ava-log-container {
            height: 100px !important;
        }
        .js-plotly-plot { /* Targets Plotly charts */
            height: 180px !important;
        }
    }
    @media (max-width: 480px) { /* Mobile phones */
        .ava-log-container {
            height: 150px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Affichage des deux images
col1, col2, col3 = st.columns(3)
with col1:
    if os.path.exists(image1):
        st.image(image1, use_container_width=True)
    else:
        st.warning(f"Image non trouvée : {image1}")
with col2:
    if os.path.exists(image2):
        st.image(image2, use_container_width=True)
    else:
        st.warning(f"Image non trouvée : {image2}")
with col3:
    if os.path.exists(image3):
        st.image(image3, use_container_width=True)
    else:
        st.warning(f"Image non trouvée : {image3}")        

st.caption("✨ Projet 3 I3AFD – Analyse vidéo par swarm multi-agents")