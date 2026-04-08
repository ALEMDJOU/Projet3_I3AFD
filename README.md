# AVA Pro — Agentic Video Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.com) *(optionnel si déployé)*  
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1.0-blue)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AVA Pro** is an **agentic system** built with LangGraph that analyzes YouTube videos by retrieving comments, filtering spam, performing multi‑dimensional analysis (sentiment, pedagogical value, recurring themes), and producing a final quality score out of 10. It supports **automatic fallback** between Gemini, Qwen 2.5, and DeepSeek LLMs.

![AVA Pro Dashboard](screenshot.png) *(add a screenshot later)*

---

## 🚀 Features

- 🔍 **YouTube search** by keywords or direct video ID / URL  
- 📊 **300 comments** retrieved (3 API pages of 100) with pagination  
- 🧹 **Advanced spam filtering** (patterns, emoji ratio, duplicates, short comments)  
- 🤖 **LangGraph multi‑agent pipeline** :  
  - `search` → `metadata` → `fetch` → `filter` → `analyst` → `synthesizer`  
- 🧠 **LLM analysis** in 7 dimensions :  
  - query/content similarity, sentiment distribution, pedagogical validation, top themes, perceived utility, categorisation, alert signals  
- 📈 **Metrics displayed** :  
  - Global score (gauge chart)  
  - Polarity (-1..1)  
  - Instructive percentage  
  - Confidence level (high/medium/low)  
  - Recommendation (Watch / According to your interests / Avoid)  
  - Sentiment timeline  
- 🔄 **Fallback chain** : Gemini → Qwen 2.5 → DeepSeek (via HuggingFace)  
- 🎨 **Modern Streamlit UI** with live logs, step tracking, and responsive design  

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ava-pro.git
cd ava-pro