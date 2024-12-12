import streamlit as st
import pandas as pd
from database import (
    init_db, save_training_run, get_training_runs,
    save_dataset, get_datasets
)
from models import ModelTrainer
from visualizations import (
    create_metrics_comparison,
    create_hyperparameter_correlation,
    create_dataset_summary
)
from utils import display_run_details, format_size
import json

# Initialize database
init_db()

# Page configuration
st.set_page_config(
    page_title="Fine-Tuning Optimization Dashboard",
    page_icon="📊",
    layout="wide"
)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Training Runs", "Datasets", "New Training Run"]
)

if page == "Dashboard":
    st.title("Fine-Tuning Optimization Dashboard")
    
    # Get data
    runs = get_training_runs()
    datasets = get_datasets()
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Runs", len(runs))
    with col2:
        st.metric("Total Datasets", len(datasets))
    with col3:
        if runs:
            latest_run = runs[0]
            st.metric("Latest Run Loss", 
                     f"{json.loads(latest_run['metrics'])['train_loss']:.4f}")
    
    # Visualizations
    st.subheader("Training Metrics")
    if runs:
        fig_loss, fig_metrics = create_metrics_comparison(runs)
        st.plotly_chart(fig_loss, use_container_width=True)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        st.subheader("Hyperparameter Analysis")
        fig_hp = create_hyperparameter_correlation(runs)
        st.plotly_chart(fig_hp, use_container_width=True)

elif page == "Training Runs":
    st.title("Training Runs")
    
    runs = get_training_runs()
    if runs:
        # Filter options
        model_filter = st.multiselect(
            "Filter by Model",
            options=list(set(run['model_name'] for run in runs))
        )
        
        filtered_runs = runs
        if model_filter:
            filtered_runs = [run for run in runs if run['model_name'] in model_filter]
        
        # Display runs
        for run in filtered_runs:
            with st.expander(f"Run {run['run_id']} - {run['model_name']}"):
                display_run_details(run)
    else:
        st.info("No training runs available yet.")

elif page == "Datasets":
    st.title("Dataset Management")
    
    # Add new dataset
    with st.expander("Add New Dataset"):
        with st.form("new_dataset"):
            name = st.text_input("Dataset Name")
            description = st.text_area("Description")
            size = st.number_input("Number of Examples", min_value=1)
            
            if st.form_submit_button("Add Dataset"):
                save_dataset(name, description, size)
                st.success("Dataset added successfully!")
    
    # Display existing datasets
    datasets = get_datasets()
    if datasets:
        st.subheader("Available Datasets")
        fig_datasets = create_dataset_summary(datasets)
        st.plotly_chart(fig_datasets, use_container_width=True)
        
        for dataset in datasets:
            with st.expander(f"{dataset['name']}"):
                st.write("Description:", dataset['description'])
                st.write("Size:", format_size(dataset['size']))
                st.write("Added:", dataset['created_at'])
    else:
        st.info("No datasets available yet.")

elif page == "New Training Run":
    st.title("Start New Training Run")
    
    with st.form("new_training"):
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            ["gpt2", "bert-base-uncased", "distilbert-base-uncased"]
        )
        
        # Dataset selection
        datasets = get_datasets()
        dataset_name = st.selectbox(
            "Select Dataset",
            [dataset['name'] for dataset in datasets]
        )
        
        # Hyperparameters
        st.subheader("Hyperparameters")
        col1, col2 = st.columns(2)
        with col1:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-6,
                max_value=1e-2,
                value=5e-5,
                format="%.1e"
            )
            num_epochs = st.number_input(
                "Number of Epochs",
                min_value=1,
                max_value=100,
                value=3
            )
        
        with col2:
            batch_size = st.selectbox(
                "Batch Size",
                [8, 16, 32, 64]
            )
            weight_decay = st.number_input(
                "Weight Decay",
                min_value=0.0,
                max_value=1.0,
                value=0.01
            )
        
        if st.form_submit_button("Start Training"):
            with st.spinner("Training in progress..."):
                trainer = ModelTrainer(model_name)
                dataset = trainer.prepare_dataset(dataset_name)
                
                hyperparameters = {
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "weight_decay": weight_decay
                }
                
                metrics = trainer.train(dataset, hyperparameters)
                
                # Save results
                save_training_run(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    hyperparameters=hyperparameters,
                    metrics=metrics
                )
                
                st.success("Training completed successfully!")
                st.write("Training Metrics:")
                st.dataframe(pd.DataFrame([metrics]))
