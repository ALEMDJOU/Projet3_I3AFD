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
    st.markdown(svg, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# En-tête principal
# ──────────────────────────────────────────────

st.markdown(
    '<div class="main-header">'
    '<h1 style="margin:0;font-size:2.5rem;font-weight:800;">Tableau de Bord AVA PRO</h1>'
    '<p style="margin:5px 0 0 0;opacity:0.9;font-size:1.1rem;">'
    'Analyse sémantique par swarm multi-agents · LangGraph + Gemini + HuggingFace Metrics'
    '</p></div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Saisie vidéo
# ──────────────────────────────────────────────

st.markdown(
    "<div class='section-title'><i class='fa-solid fa-link'></i> Analyser une vidéo</div>",
    unsafe_allow_html=True,
)
col_input, col_btn = st.columns([4, 1.2])
with col_input:
    video_input = st.text_input(
        "URL ou ID YouTube",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
with col_btn:
    run_button = st.button("🚀 LANCER L'ANALYSE", type="primary")

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
            " Suivi de l'Exécution</div>",
            unsafe_allow_html=True,
        )
        col_exec, col_logs = st.columns([2, 1])
        with col_exec:
            tracker_box  = st.empty()
            progress_bar = st.progress(0)
            if st.button("🛑 ARRÊTER L'ANALYSE", type="secondary", use_container_width=True):
                st.rerun()
        with col_logs:
            st.markdown("<h4 style='font-size:0.9rem;color:#64748B;'>LOGS</h4>",
                        unsafe_allow_html=True)
            log_container = st.empty()

        steps_display = [
            ("fetcher",     "Fetcher",     "fa-download"),
            ("filter",      "Filter",      "fa-filter"),
            ("analyst",     "Analyst",     "fa-brain"),
            ("synthesizer", "Synthesizer", "fa-wand-magic-sparkles"),
        ]
        completed = []

        with st.spinner("Activation du swarm…"):
            for i, output in enumerate(app_graph.stream({"video_id": video_id}, config)):
                active_node = list(output.keys())[0]
                completed.append(active_node)

                state = app_graph.get_state(config).values
                logs  = state.get("reflection_logs", [])

                # Logs
                log_html = ("<div style='height:150px;overflow-y:auto;font-size:0.8rem;"
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

        st.success("✅ Analyse terminée.")

        # ──────────────────────────────────────────────
        # Affichage des résultats
        # ──────────────────────────────────────────────

        final_state = app_graph.get_state(config).values
        info  = final_state.get("video_info", {"title": "N/A", "author": "N/A"})
        score = final_state.get("final_score", 0)

        st.markdown("---")
        st.markdown(
            "<div class='section-title'><i class='fa-solid fa-chart-pie'></i>"
            " Résultats de l'Analyse</div>",
            unsafe_allow_html=True,
        )

        col_gauge, col_trend, col_info = st.columns([1.5, 2, 2.5])

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "QUALITÉ GLOBALE", "font": {"size": 16, "color": "#018EA9"}},
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
                    title={"text": "ÉVOLUTION DU SENTIMENT",
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
                .replace("RÉSUMÉ:", "")
                .strip()
            )
            # Carte Métadonnées
            st.markdown(
                f'<div class="info-card">'
                f'<h4 style="color:#64748B;margin:0;font-size:0.75rem;letter-spacing:1px;">SOURCE</h4>'
                f'<p style="font-weight:700;color:#1E293B;font-size:1.05rem;margin:5px 0;">'
                f'{info["title"]}</p>'
                f'<p style="color:#00D4AC;font-size:0.9rem;margin:0;">'
                f'<i class="fa-solid fa-circle-user"></i> {info["author"]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Bulle de commentaire de l'Analyste
            st.markdown(
                f'<div class="comment-box">'
                f'  <div class="analyst-avatar"><i class="fa-solid fa-brain"></i></div>'
                f'  <div class="comment-content">'
                f'    <div class="comment-author">AVA Analyst <span class="author-badge">Expert Swarm</span></div>'
                f'    <div class="comment-text">"{summary_text}"</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Tabs commentaires
        tab1, tab2, tab3 = st.tabs(["Bruts", "Filtrés", "Analyse"])
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
        render_ablation(final_state, None, keys["gemini"])
