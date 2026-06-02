#!/bin/bash
# Streamlit デモアプリ起動スクリプト
cd "$(dirname "$0")/.."
.venv/bin/streamlit run app/streamlit_app.py --server.port=8501 --server.headless=false
