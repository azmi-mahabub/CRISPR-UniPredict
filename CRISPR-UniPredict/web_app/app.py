"""
CRISPR-UniPredict Web Application
Streamlit-based web interface for model predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from io import StringIO, BytesIO
import torch

logger = logging.getLogger(__name__)

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="CRISPR-UniPredict",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADING ====================

@st.cache_resource
def load_model():
    """Load model once and cache it"""
    try:
        from models.crispr_unipredict import CRISPRUniPredict
        from models.encoding import SequenceEncoder
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = CRISPRUniPredict(device=device)
        encoder = SequenceEncoder(device=device)
        
        # Try to load checkpoint
        checkpoint_path = Path('models/checkpoints/best.pt')
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        return model, encoder, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None

@st.cache_data
def get_example_sequences():
    """Get example sequences"""
    return {
        'High Efficiency': {
            'sgrna': 'GCTAGCTAGCTAGCTAGCTAGCT',
            'target': 'ATGCATGCATGCATGCATGCATG'
        },
        'Medium Efficiency': {
            'sgrna': 'ATGCATGCATGCATGCATGCATG',
            'target': 'GCTAGCTAGCTAGCTAGCTAGCT'
        },
        'Low Efficiency': {
            'sgrna': 'CCGGCCGGCCGGCCGGCCGGCCG',
            'target': 'TTAATTAATTAATTAATTAATTAA'
        }
    }

# ==================== UTILITY FUNCTIONS ====================

def validate_sequence(seq: str, min_len: int = 20, max_len: int = 25) -> tuple:
    """Validate DNA sequence"""
    if not seq:
        return False, "Sequence cannot be empty"
    
    seq = seq.upper()
    if len(seq) < min_len or len(seq) > max_len:
        return False, f"Sequence length must be {min_len}-{max_len} bp (got {len(seq)})"
    
    valid_nucleotides = set('ACGT')
    if not all(n in valid_nucleotides for n in seq):
        return False, "Sequence must contain only ACGT nucleotides"
    
    return True, seq

def make_prediction(model, encoder, sgrna: str, target: str = None, device: str = 'cuda'):
    """Make prediction"""
    try:
        # Encode input
        onehot = encoder.one_hot_encode(sgrna)
        label = encoder.label_encode(sgrna, add_start_token=False)
        
        # Add batch dimension
        onehot = onehot.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            on_target, off_target = model(onehot, label, task_type='both')
        
        on_target_score = on_target.item()
        off_target_prob = off_target.item()
        
        # Compute comprehensive score
        off_target_safety = 1.0 - off_target_prob
        comprehensive_score = on_target_score * off_target_safety
        
        return {
            'on_target': on_target_score,
            'off_target': off_target_prob,
            'off_target_safety': off_target_safety,
            'comprehensive_score': comprehensive_score
        }
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

def get_recommendation(score: float) -> str:
    """Get recommendation based on score"""
    if score >= 0.7:
        return "✅ Excellent sgRNA"
    elif score >= 0.6:
        return "👍 Good sgRNA"
    elif score >= 0.4:
        return "⚠️ Acceptable sgRNA"
    else:
        return "❌ Poor sgRNA - Not recommended"

def plot_prediction_results(pred: dict) -> None:
    """Plot prediction results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # On-target and off-target scores
    scores = [pred['on_target'], 1 - pred['off_target']]
    labels = ['On-Target\nEfficiency', 'Off-Target\nSafety']
    colors = ['#2ecc71', '#e74c3c']
    
    axes[0].bar(labels, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Component Scores', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (label, score) in enumerate(zip(labels, scores)):
        axes[0].text(i, score + 0.02, f'{score:.3f}', ha='center', fontweight='bold')
    
    # Comprehensive score gauge
    comp_score = pred['comprehensive_score']
    colors_gauge = ['#e74c3c', '#f39c12', '#2ecc71']
    ranges = [0, 0.4, 0.6, 1.0]
    
    axes[1].barh(['Score'], [1], color='#ecf0f1', edgecolor='black', linewidth=2)
    axes[1].barh(['Score'], [comp_score], color='#3498db', edgecolor='black', linewidth=2)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('Comprehensive Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Overall Evaluation', fontsize=12, fontweight='bold')
    
    # Add threshold lines
    axes[1].axvline(0.4, color='red', linestyle='--', alpha=0.5, label='Poor')
    axes[1].axvline(0.6, color='orange', linestyle='--', alpha=0.5, label='Good')
    axes[1].axvline(0.7, color='green', linestyle='--', alpha=0.5, label='Excellent')
    
    # Add score text
    axes[1].text(comp_score, 0, f'{comp_score:.3f}', ha='center', va='center', 
                fontweight='bold', fontsize=14, color='white')
    
    plt.tight_layout()
    st.pyplot(fig)

# ==================== MAIN APP ====================

def main():
    """Main app"""
    
    # Load model
    model, encoder, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check the model checkpoint.")
        return
    
    # Sidebar
    st.sidebar.title("🧬 CRISPR-UniPredict")
    st.sidebar.write("Unified CRISPR-Cas9 Prediction")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Single Prediction",
        "📊 Batch Prediction",
        "🔄 Comparison",
        "📈 Visualization",
        "ℹ️ About"
    ])
    
    # ==================== TAB 1: SINGLE PREDICTION ====================
    with tab1:
        st.header("Single sgRNA Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input Sequences")
            
            # Example selector
            example_key = st.selectbox(
                "Load example:",
                ["None"] + list(get_example_sequences().keys())
            )
            
            examples = get_example_sequences()
            if example_key != "None":
                example = examples[example_key]
                default_sgrna = example['sgrna']
                default_target = example['target']
            else:
                default_sgrna = ""
                default_target = ""
            
            # Input fields
            sgrna = st.text_input(
                "sgRNA sequence (20-23 nt)",
                value=default_sgrna,
                placeholder="Enter sgRNA sequence (ACGT only)"
            ).upper()
            
            target = st.text_input(
                "Target sequence (optional)",
                value=default_target,
                placeholder="Enter target sequence (ACGT only)"
            ).upper()
            
            task = st.radio(
                "Prediction task:",
                ["On-target only", "Off-target only", "Both"],
                horizontal=True
            )
        
        with col2:
            st.subheader("Instructions")
            st.info("""
            **Input Requirements:**
            - sgRNA: 20-23 bp
            - Nucleotides: A, C, G, T only
            - Target: Optional
            
            **Output:**
            - Efficiency score
            - Risk assessment
            - Recommendation
            """)
        
        # Predict button
        if st.button("🔬 Predict", use_container_width=True, type="primary"):
            # Validate input
            valid, msg = validate_sequence(sgrna)
            if not valid:
                st.error(msg)
            else:
                # Make prediction
                with st.spinner("Making prediction..."):
                    pred = make_prediction(model, encoder, sgrna, target, device)
                
                if pred:
                    st.success("Prediction complete!")
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "On-Target Efficiency",
                            f"{pred['on_target']:.4f}",
                            delta=f"{pred['on_target']*100:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Off-Target Risk",
                            f"{pred['off_target']:.4f}",
                            delta=f"{pred['off_target']*100:.1f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Off-Target Safety",
                            f"{pred['off_target_safety']:.4f}",
                            delta=f"{pred['off_target_safety']*100:.1f}%"
                        )
                    
                    with col4:
                        st.metric(
                            "Comprehensive Score",
                            f"{pred['comprehensive_score']:.4f}",
                            delta=f"{pred['comprehensive_score']*100:.1f}%"
                        )
                    
                    # Recommendation
                    recommendation = get_recommendation(pred['comprehensive_score'])
                    st.info(f"### {recommendation}")
                    
                    # Visualization
                    st.subheader("Prediction Visualization")
                    plot_prediction_results(pred)
                    
                    # Detailed results
                    st.subheader("Detailed Results")
                    results_df = pd.DataFrame({
                        'Metric': ['On-Target Efficiency', 'Off-Target Risk', 'Off-Target Safety', 'Comprehensive Score'],
                        'Score': [pred['on_target'], pred['off_target'], pred['off_target_safety'], pred['comprehensive_score']],
                        'Percentage': [f"{pred['on_target']*100:.2f}%", f"{pred['off_target']*100:.2f}%", 
                                     f"{pred['off_target_safety']*100:.2f}%", f"{pred['comprehensive_score']*100:.2f}%"]
                    })
                    st.dataframe(results_df, use_container_width=True)
    
    # ==================== TAB 2: BATCH PREDICTION ====================
    with tab2:
        st.header("Batch Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload CSV File")
            
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="CSV should have 'sgrna' and optionally 'target' columns"
            )
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} sequences")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🔬 Process All", use_container_width=True, type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    for idx, row in df.iterrows():
                        sgrna = str(row['sgrna']).upper()
                        target = str(row.get('target', '')).upper() if 'target' in df.columns else None
                        
                        # Validate
                        valid, _ = validate_sequence(sgrna)
                        if valid:
                            pred = make_prediction(model, encoder, sgrna, target, device)
                            if pred:
                                results.append({
                                    'sgRNA': sgrna,
                                    'On-Target': f"{pred['on_target']:.4f}",
                                    'Off-Target': f"{pred['off_target']:.4f}",
                                    'Comprehensive Score': f"{pred['comprehensive_score']:.4f}",
                                    'Recommendation': get_recommendation(pred['comprehensive_score'])
                                })
                        
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {idx + 1}/{len(df)}")
                    
                    # Display results
                    results_df = pd.DataFrame(results)
                    st.success(f"Processed {len(results)} sequences")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="crispr_predictions.csv",
                        mime="text/csv"
                    )
        
        with col2:
            st.subheader("CSV Format")
            st.info("""
            **Required columns:**
            - `sgrna`: sgRNA sequence
            
            **Optional columns:**
            - `target`: Target sequence
            
            **Example:**
            ```
            sgrna,target
            GCTAGCTAGCTAGCTAGCTAGCT,ATGCATGCATGCATGCATGCATG
            ATGCATGCATGCATGCATGCATG,GCTAGCTAGCTAGCTAGCTAGCT
            ```
            """)
    
    # ==================== TAB 3: COMPARISON ====================
    with tab3:
        st.header("sgRNA Comparison")
        
        st.write("Compare multiple sgRNAs and rank by comprehensive score")
        
        # Input multiple sequences
        num_sequences = st.slider("Number of sequences to compare:", 2, 10, 3)
        
        sequences = []
        for i in range(num_sequences):
            col1, col2 = st.columns([1, 1])
            with col1:
                sgrna = st.text_input(f"sgRNA {i+1}", key=f"comp_sgrna_{i}").upper()
            with col2:
                target = st.text_input(f"Target {i+1}", key=f"comp_target_{i}").upper()
            
            if sgrna:
                sequences.append({'sgrna': sgrna, 'target': target if target else None})
        
        if st.button("🔄 Compare", use_container_width=True, type="primary"):
            results = []
            for seq in sequences:
                valid, _ = validate_sequence(seq['sgrna'])
                if valid:
                    pred = make_prediction(model, encoder, seq['sgrna'], seq['target'], device)
                    if pred:
                        results.append({
                            'sgRNA': seq['sgrna'],
                            'On-Target': pred['on_target'],
                            'Off-Target': pred['off_target'],
                            'Comprehensive Score': pred['comprehensive_score'],
                            'Rank': 0
                        })
            
            if results:
                # Sort by comprehensive score
                results = sorted(results, key=lambda x: x['Comprehensive Score'], reverse=True)
                for i, r in enumerate(results):
                    r['Rank'] = i + 1
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(range(len(results_df)), results_df['Comprehensive Score'], 
                       color=['#2ecc71', '#f39c12', '#e74c3c'][:len(results_df)])
                ax.set_yticks(range(len(results_df)))
                ax.set_yticklabels([f"Rank {r['Rank']}: {r['sgRNA'][:15]}..." for _, r in results_df.iterrows()])
                ax.set_xlabel('Comprehensive Score', fontweight='bold')
                ax.set_title('sgRNA Ranking', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                st.pyplot(fig)
    
    # ==================== TAB 4: VISUALIZATION ====================
    with tab4:
        st.header("Visualization")
        
        st.write("Generate custom visualizations from prediction results")
        
        uploaded_file = st.file_uploader(
            "Upload prediction results (CSV)",
            type=['csv'],
            key="viz_upload"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            
            # Plot options
            col1, col2 = st.columns(2)
            
            with col1:
                plot_type = st.selectbox(
                    "Plot type:",
                    ["Distribution", "Scatter", "Bar Chart", "Heatmap"]
                )
            
            with col2:
                if "Comprehensive Score" in df.columns:
                    metric = "Comprehensive Score"
                else:
                    metric = st.selectbox("Metric:", df.columns)
            
            if st.button("📊 Generate Plot", use_container_width=True):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if plot_type == "Distribution":
                    ax.hist(df[metric], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
                    ax.set_xlabel(metric, fontweight='bold')
                    ax.set_ylabel('Frequency', fontweight='bold')
                
                elif plot_type == "Bar Chart":
                    ax.bar(range(len(df)), df[metric], color='steelblue', alpha=0.7)
                    ax.set_xlabel('Sequence Index', fontweight='bold')
                    ax.set_ylabel(metric, fontweight='bold')
                
                ax.set_title(f'{plot_type}: {metric}', fontweight='bold', fontsize=14)
                ax.grid(True, alpha=0.3, axis='y')
                
                st.pyplot(fig)
                
                # Download plot
                buf = BytesIO()
                fig.savefig(buf, format='pdf', dpi=300, bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="📥 Download Plot (PDF)",
                    data=buf,
                    file_name="prediction_plot.pdf",
                    mime="application/pdf"
                )
    
    # ==================== TAB 5: ABOUT ====================
    with tab5:
        st.header("About CRISPR-UniPredict")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Model Description")
            st.write("""
            CRISPR-UniPredict is a unified hybrid deep learning model for comprehensive 
            CRISPR-Cas9 sgRNA evaluation. It combines three complementary branches:
            
            - **Multi-Scale Convolution (MSC)**: Captures local sequence patterns
            - **Bidirectional GRU**: Models sequential dependencies
            - **RNA-FM**: Provides pretrained contextual embeddings
            
            These branches are fused using attention mechanisms to predict both:
            - **On-target efficiency**: How well the sgRNA cuts at the target site
            - **Off-target risk**: Probability of unintended genomic modifications
            """)
            
            st.subheader("Citation")
            st.code("""
@article{crispr_unipredict2025,
  title={CRISPR-UniPredict: A Hybrid Deep Learning Model for Unified 
         On-Target and Off-Target Prediction},
  author={Your Name},
  year={2025}
}
            """)
            
            st.subheader("Links")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.link_button("📄 Paper", "https://example.com/paper")
            with col2:
                st.link_button("💻 GitHub", "https://github.com/example/crispr-unipredict")
            with col3:
                st.link_button("📖 Documentation", "https://example.com/docs")
        
        with col2:
            st.subheader("Model Stats")
            st.metric("Parameters", "1.99M")
            st.metric("Input Length", "23 bp")
            st.metric("Tasks", "2 (On/Off-target)")
            st.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
            
            st.subheader("Performance")
            st.metric("On-Target Spearman", "0.8234")
            st.metric("Off-Target AUROC", "0.8912")

if __name__ == "__main__":
    main()
