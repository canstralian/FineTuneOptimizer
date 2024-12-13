
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict

def create_metrics_comparison(runs: List[Dict]):
    """Create comparison plots for different training runs"""
    if not runs:
        return None, None
        
    df = pd.DataFrame(runs)
    metrics_df = pd.json_normalize([run['metrics'] for run in runs])
    df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
    
    # Training loss over time
    fig_loss = px.line(df, 
                       x='created_at', 
                       y='train_loss',
                       color='model_name',
                       title='Training Loss Over Time')
    
    # Performance metrics comparison
    fig_metrics = None
    if 'eval_perplexity' in df.columns:
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Bar(x=df['created_at'], 
                                   y=df['eval_perplexity'],
                                   name='Perplexity'))
        fig_metrics.update_layout(title='Model Performance Metrics')
    
    return fig_loss, fig_metrics

def create_hyperparameter_correlation(runs: List[Dict]):
    """Create correlation plot between hyperparameters and metrics"""
    if not runs:
        return None
        
    df = pd.DataFrame(runs)
    hp_df = pd.json_normalize([run['hyperparameters'] for run in runs])
    metrics_df = pd.json_normalize([run['metrics'] for run in runs])
    df = pd.concat([df.drop(['hyperparameters', 'metrics'], axis=1), 
                   hp_df, 
                   metrics_df], axis=1)
    
    fig = px.scatter(df, 
                    x='learning_rate',
                    y='train_loss',
                    color='model_name',
                    size='batch_size',
                    title='Hyperparameter Impact on Training Loss')
    
    return fig

def create_dataset_summary(datasets: List[Dict]):
    """Create summary visualization for datasets"""
    if not datasets:
        return None
        
    df = pd.DataFrame(datasets)
    
    fig = px.bar(df,
                 x='name',
                 y='size',
                 title='Dataset Sizes',
                 labels={'size': 'Number of Examples'})
    
    return fig
