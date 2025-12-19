"""
Streamlit web interface for testing and comparing RAG models.
"""
import streamlit as st
import pandas as pd
import torch
import yaml
import os
from pathlib import Path
from src.model_loader import load_model_with_adapters
from src.rag_pipeline import build_rag_pipeline
from src.data_loader import load_and_preprocess_data
from src.enhanced_evaluator import format_prompt
from transformers import AutoTokenizer
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="RAG Model Testing Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_experiment_results():
    """Load results from CSV."""
    if os.path.exists('results.csv'):
        return pd.read_csv('results.csv')
    return pd.DataFrame()

@st.cache_resource
def load_config():
    """Load configuration."""
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    return None

def load_model_for_inference(experiment_name, config):
    """Load a trained model for inference."""
    # Find experiment config
    experiments = config.get('experiments', [])
    exp_config = None
    for exp in experiments:
        if exp.get('experiment_name') == experiment_name:
            exp_config = exp
            break
    
    if not exp_config:
        return None, None, None
    
    # Load model
    try:
        model, tokenizer, _ = load_model_with_adapters(
            base_model_name=exp_config['model']['base_model'],
            adapter_type=exp_config['model'].get('adapter_type', 'none'),
            adapter_config=exp_config['model'],
            total_training_steps=None
        )
        return model, tokenizer, exp_config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_resource
def load_rag_pipeline():
    """Load RAG pipeline."""
    config = load_config()
    if not config:
        return None
    
    global_settings = config.get('global_settings', {})
    dataset = load_and_preprocess_data(
        global_settings['data']['dataset_name'],
        global_settings['data']['split']
    )
    knowledge_base_texts = [item['context'] for item in dataset]
    
    rag_config = global_settings.get('rag', {})
    
    # Use advanced RAG if enabled
    if rag_config.get('enable_reranking', False) or rag_config.get('enable_sparse', False):
        from src.advanced_rag import build_advanced_rag_pipeline
        retriever = build_advanced_rag_pipeline(
            knowledge_base_texts,
            model_name=rag_config.get('retriever_model', 'all-MiniLM-L6-v2'),
            enable_reranking=rag_config.get('enable_reranking', False),
            enable_sparse=rag_config.get('enable_sparse', False)
        )
    else:
        from src.rag_pipeline import build_rag_pipeline
        retriever = build_rag_pipeline(
            knowledge_base_texts,
            model_name=rag_config.get('retriever_model', 'all-MiniLM-L6-v2')
        )
    
    return retriever, dataset

def main():
    st.markdown('<div class="main-header">🤖 RAG Model Testing Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load data
        results_df = load_experiment_results()
        config = load_config()
        
        if results_df.empty:
            st.warning("No results found. Please run training first.")
            return
        
        # Model selection
        st.subheader("📊 View Results")
        view_mode = st.radio(
            "Select View",
            ["📈 Results Overview", "🧪 Test Models", "🔍 Compare Models"]
        )
        
        if view_mode == "🧪 Test Models":
            st.subheader("🎯 Model Selection")
            experiment_names = results_df['experiment_name'].tolist()
            selected_experiment = st.selectbox(
                "Choose a model to test",
                experiment_names,
                help="Select a trained model to test with custom queries"
            )
            
            st.subheader("⚙️ Inference Settings")
            top_k = st.slider("Top K contexts", 1, 10, 3)
            max_tokens = st.slider("Max tokens", 10, 200, 50)
            temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
    
    # Main content
    if view_mode == "📈 Results Overview":
        show_results_overview(results_df)
    elif view_mode == "🧪 Test Models":
        show_model_testing(selected_experiment, config, top_k, max_tokens, temperature)
    elif view_mode == "🔍 Compare Models":
        show_model_comparison(results_df)

def show_results_overview(results_df):
    """Display results overview with visualizations."""
    st.header("📊 Experiment Results Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Experiments", len(results_df))
    with col2:
        best_semantic = results_df['semantic_accuracy'].max()
        st.metric("Best Semantic Accuracy", f"{best_semantic:.3f}")
    with col3:
        best_rouge = results_df['rougeL'].max()
        st.metric("Best ROUGE-L", f"{best_rouge:.3f}")
    with col4:
        avg_time = results_df['training_time_sec'].mean()
        st.metric("Avg Training Time", f"{avg_time:.1f}s")
    
    # Results table
    st.subheader("📋 Detailed Results")
    st.dataframe(results_df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance vs Efficiency")
        fig = px.scatter(
            results_df,
            x='trainable_params',
            y='semantic_accuracy',
            color='adapter_type',
            size='training_time_sec',
            hover_data=['experiment_name'],
            log_x=True,
            title="Semantic Accuracy vs Trainable Parameters"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ROUGE-L Scores")
        fig = px.bar(
            results_df.sort_values('rougeL', ascending=True),
            x='rougeL',
            y='experiment_name',
            orientation='h',
            title="ROUGE-L Score by Experiment"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Metrics comparison
    st.subheader("📊 Metrics Comparison")
    metric_cols = ['rougeL', 'rouge1', 'rouge2', 'bleu', 'semantic_accuracy', 'f1_score']
    fig = go.Figure()
    
    for metric in metric_cols:
        if metric in results_df.columns:
            fig.add_trace(go.Scatter(
                x=results_df['experiment_name'],
                y=results_df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="All Metrics Comparison",
        xaxis_title="Experiment",
        yaxis_title="Score",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

def show_model_testing(experiment_name, config, top_k, max_tokens, temperature):
    """Show model testing interface."""
    st.header(f"🧪 Testing: {experiment_name}")
    
    # Load RAG pipeline
    with st.spinner("Loading RAG pipeline..."):
        rag_data = load_rag_pipeline()
        if not rag_data:
            st.error("Failed to load RAG pipeline")
            return
        retriever, dataset = rag_data
    
    # Load model
    with st.spinner(f"Loading model {experiment_name}..."):
        model, tokenizer, exp_config = load_model_for_inference(experiment_name, config)
        if not model:
            st.error("Failed to load model")
            return
    
    st.success("✅ Model loaded successfully!")
    
    # Query input
    st.subheader("💬 Enter Your Question")
    
    # Retrieval strategy selector (if advanced RAG is available)
    from src.advanced_rag import AdvancedRAGRetriever
    if isinstance(retriever, AdvancedRAGRetriever):
        retrieval_strategy = st.selectbox(
            "Retrieval Strategy",
            ["dense", "sparse", "hybrid", "reranked"],
            index=0,
            help="Choose retrieval strategy (only for advanced RAG)"
        )
        st.session_state['retrieval_strategy'] = retrieval_strategy
    
    query = st.text_area(
        "Question",
        placeholder="Enter your question here...",
        height=100
    )
    
    # Example questions
    with st.expander("📚 Example Questions"):
        example_questions = [
            item['question'] for item in dataset.select(range(10))
        ]
        for i, q in enumerate(example_questions, 1):
            if st.button(f"Example {i}: {q[:60]}...", key=f"example_{i}"):
                query = q
                st.rerun()
    
    if st.button("🚀 Get Answer", type="primary"):
        if not query:
            st.warning("Please enter a question")
            return
        
        with st.spinner("Generating answer..."):
            # Retrieve context (support advanced RAG)
            from src.advanced_rag import AdvancedRAGRetriever
            if isinstance(retriever, AdvancedRAGRetriever):
                strategy = st.session_state.get('retrieval_strategy', 'dense')
                retrieved_contexts = retriever.retrieve(query, top_k=top_k, strategy=strategy)
            else:
                retrieved_contexts = retriever.retrieve(query, top_k=top_k)
            retrieved_context = "\n".join(retrieved_contexts)
            prompt = format_prompt(query, retrieved_context)
            
            # Generate answer
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output_sequences = model.generate(
                    input_ids=inputs['input_ids'],
                    max_new_tokens=max_tokens,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                )
            
            generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            answer = generated_text.split("Answer:")[-1].strip()
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Answer")
            st.info(answer)
            
            st.subheader("📄 Retrieved Context")
            for i, ctx in enumerate(retrieved_contexts, 1):
                with st.expander(f"Context {i}"):
                    st.text(ctx[:500] + "..." if len(ctx) > 500 else ctx)
        
        with col2:
            st.subheader("📊 Information")
            st.metric("Retrieved Contexts", len(retrieved_contexts))
            st.metric("Answer Length", len(answer.split()))
            st.metric("Model", exp_config['model']['base_model'])
            st.metric("Adapter", exp_config['model'].get('adapter_type', 'Full FT'))

def show_model_comparison(results_df):
    """Show model comparison interface."""
    st.header("🔍 Model Comparison")
    
    # Select models to compare
    experiment_names = results_df['experiment_name'].tolist()
    selected_models = st.multiselect(
        "Select models to compare",
        experiment_names,
        default=experiment_names[:3] if len(experiment_names) >= 3 else experiment_names
    )
    
    if not selected_models:
        st.warning("Please select at least one model")
        return
    
    comparison_df = results_df[results_df['experiment_name'].isin(selected_models)]
    
    # Comparison table
    st.subheader("📊 Side-by-Side Comparison")
    
    # Key metrics
    metrics_to_show = ['experiment_name', 'adapter_type', 'trainable_percent', 
                       'training_time_sec', 'rougeL', 'rouge1', 'rouge2', 
                       'bleu', 'semantic_accuracy', 'f1_score']
    available_metrics = [m for m in metrics_to_show if m in comparison_df.columns]
    
    st.dataframe(
        comparison_df[available_metrics].set_index('experiment_name'),
        use_container_width=True
    )
    
    # Visual comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Semantic Accuracy")
        fig = px.bar(
            comparison_df,
            x='experiment_name',
            y='semantic_accuracy',
            title="Semantic Accuracy Comparison"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Training Time")
        fig = px.bar(
            comparison_df,
            x='experiment_name',
            y='training_time_sec',
            title="Training Time Comparison"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart for multi-metric comparison
    if len(selected_models) <= 5:  # Limit for readability
        st.subheader("📈 Multi-Metric Radar Chart")
        
        metrics_for_radar = ['rougeL', 'rouge1', 'semantic_accuracy', 'f1_score']
        available_radar_metrics = [m for m in metrics_for_radar if m in comparison_df.columns]
        
        if available_radar_metrics:
            fig = go.Figure()
            
            for _, row in comparison_df.iterrows():
                values = [row[m] for m in available_radar_metrics]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=available_radar_metrics,
                    fill='toself',
                    name=row['experiment_name']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Multi-Metric Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

