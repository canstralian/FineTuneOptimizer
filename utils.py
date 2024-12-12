import streamlit as st
import pandas as pd
from typing import Dict, Any

def format_metrics(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Format metrics for display"""
    df = pd.DataFrame([metrics])
    df = df.round(4)
    return df

def display_run_details(run: Dict[str, Any]):
    """Display detailed information about a training run"""
    st.subheader("Run Details")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Model:", run['model_name'])
        st.write("Dataset:", run['dataset_name'])
        st.write("Created:", run['created_at'])
    
    with col2:
        st.write("Hyperparameters:")
        st.json(run['hyperparameters'])
    
    st.write("Metrics:")
    metrics_df = format_metrics(run['metrics'])
    st.dataframe(metrics_df)

def format_size(size: int) -> str:
    """Format dataset size for display"""
    if size < 1000:
        return f"{size} examples"
    elif size < 1000000:
        return f"{size/1000:.1f}K examples"
    else:
        return f"{size/1000000:.1f}M examples"
