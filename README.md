# Wen IA — Workbench de Exploración Numérica, RAG & Big Data Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![DuckDB](https://img.shields.io/badge/OLAP-DuckDB-FFF000.svg?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![LangChain](https://img.shields.io/badge/orchestration-LangChain-1C3C3C.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/vectorstore-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Plataforma fullstack de grado científico diseñada para el procesamiento multivariable de alta dimensionalidad, análisis columnar OLAP en memoria y generación aumentada por recuperación (**RAG**) con modelos de lenguaje de última generación (**LLMs**).

---

## Arquitectura General del Sistema

```text
                           ┌────────────────────────────────────────┐
                           │      Experimental Dataset Ingestion    │
                           │      (.xlsx, .xls, .ods via Streamlit) │
                           └───────────────────┬────────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
┌──────────────────────────────┐                              ┌──────────────────────────────┐
│       Analytics Engine       │                              │    Vector & RAG Engine       │
│  (Scikit-Learn, SciPy, SHAP) │                              │    (LangChain + ChromaDB)    │
├──────────────────────────────┤                              ├──────────────────────────────┤
│ • Pearson Correlation Matrix │                              │ • Ephemeral Session Stores   │
│ • 2D PCA Biplot & Loadings   │                              │ • Scikit-Learn Hash Fallback │
│ • Ward Clustering & Cophenet │                              │ • 360° Analytical Dossier    │
│ • TOPSIS Multicriteria Rank  │                              │ • Grounded Question-Answering│
│ • Isolation Forest & Mahalan.│                              └──────────────┬───────────────┘
│ • Random Forest & SHAP (XAI) │                                             │
└──────────────┬───────────────┘                                             ▼
               │                                              ┌──────────────────────────────┐
               ▼                                              │ Multi-Provider LLM Gateway   │
┌──────────────────────────────┐                              ├──────────────────────────────┤
│    In-Memory OLAP Layer      │                              │ • Dynamic Model Discovery    │
│   (DuckDB & Apache Arrow)    │                              │ • Anti-Loop Penalties        │
├──────────────────────────────┤                              │ • Groq (LLaMA 3.3/3.1)       │
│ • Sub-millisecond SQL Engine │                              │ • Google Gemini (1.5/2.0)    │
│ • Vectorized Columnar Memory │                              │ • OpenAI (GPT-4o-mini)       │
└──────────────────────────────┘                              └──────────────────────────────┘
