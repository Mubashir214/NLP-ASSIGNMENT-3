import streamlit as st
import os
import sys

# Check and install missing packages
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.info("Please make sure all packages in requirements.txt are installed")
    sys.exit(1)

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""
    try:
        model_path = "./bert-feedback-best"
        
        # Check if model directory exists
        if not os.path.exists(model_path):
            st.error(f"Model directory '{model_path}' not found!")
            return None, None
        
        # Check for essential files
        essential_files = {
            'pytorch_model.bin': 'Model weights file',
            'config.json': 'Model configuration file',
            'vocab.txt': 'Tokenizer vocabulary file',
            'tokenizer_config.json': 'Tokenizer configuration file'
        }
        
        missing_files = []
        for file, description in essential_files.items():
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                missing_files.append(f"{file} ({description})")
        
        if missing_files:
            st.error("Missing essential model files:")
            for file in missing_files:
                st.error(f"  - {file}")
            return None, None
        
        st.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        st.info("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        st.success("✅ Model loaded successfully!")
        return tokenizer, model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("This might be due to version incompatibility. Check the requirements.txt")
        return None, None

def predict_sentiment(texts, tokenizer, model, top_k=3):
    """
    Predict sentiment for a list of texts
    """
    try:
        # Get label mapping from model config
        if hasattr(model.config, 'id2label'):
            id2label = model.config.id2label
        else:
            # Fallback to default mapping
            id2label = {0: "negative", 1: "neutral", 2: "positive"}
        
        # Tokenize inputs
        encodings = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        # Move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        
        results = []
        for i, text in enumerate(texts):
            pred_idx = probabilities[i].argmax()
            pred_label = id2label.get(int(pred_idx), "unknown")
            pred_prob = float(probabilities[i, pred_idx])
            
            # Get top-k predictions
            top_indices = probabilities[i].argsort()[-top_k:][::-1]
            topk_predictions = [
                (id2label.get(int(idx), "unknown"), float(probabilities[i, idx]))
                for idx in top_indices
            ]
            
            results.append({
                "text": text,
                "predicted_label": pred_label,
                "confidence": pred_prob,
                "top_predictions": topk_predictions
            })
        
        return results
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return []

def main():
    st.title("🎯 BERT Sentiment Analysis App")
    st.write("Analyze sentiment in customer feedback using your trained BERT model")
    
    # Display system info
    st.sidebar.title("System Info")
    st.sidebar.write(f"PyTorch version: {torch.__version__}")
    st.sidebar.write(f"Transformers version: {sys.modules['transformers'].__version__}")
    st.sidebar.write(f"Device: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")
    
    # Load model
    with st.spinner("Loading model and tokenizer..."):
        tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("""
        ❌ Failed to load model. Please check:
        
        1. **Model Files**: Ensure all model files are in the `bert-feedback-best` folder:
           - `pytorch_model.bin` 
           - `config.json`
           - `vocab.txt`
           - `tokenizer_config.json`
           - `special_tokens_map.json`
        
        2. **Dependencies**: Make sure all packages are installed:
           ```bash
           pip install -r requirements.txt
           ```
        
        3. **File Structure**:
           ```
           your-app/
           ├── app.py
           ├── requirements.txt
           └── bert-feedback-best/
               ├── pytorch_model.bin
               ├── config.json
               ├── vocab.txt
               ├── tokenizer_config.json
               └── special_tokens_map.json
           ```
        """)
        return
    
    # Display model info in sidebar
    st.sidebar.title("Model Information")
    if hasattr(model.config, 'model_type'):
        st.sidebar.write(f"**Model Type:** {model.config.model_type}")
    st.sidebar.write(f"**Number of Labels:** {model.config.num_labels}")
    
    if hasattr(model.config, 'id2label'):
        st.sidebar.write("**Label Mapping:**")
        st.sidebar.json(model.config.id2label)
    else:
        st.sidebar.write("**Label Mapping:** Using default mapping")
        st.sidebar.json({0: "negative", 1: "neutral", 2: "positive"})
    
    # Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Single Text Analysis", "Batch Analysis", "Model Info"]
    )
    
    if app_mode == "Single Text Analysis":
        single_text_analysis(tokenizer, model)
    
    elif app_mode == "Batch Analysis":
        batch_analysis(tokenizer, model)
    
    elif app_mode == "Model Info":
        model_info(tokenizer, model)

def single_text_analysis(tokenizer, model):
    st.header("🔍 Single Text Analysis")
    
    # Text input
    user_input = st.text_area(
        "Enter text to analyze:",
        height=100,
        placeholder="Type your text here... (e.g., 'The product is amazing and delivery was fast!')"
    )
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        show_top_k = st.checkbox("Show top 3 predictions", value=True)
    
    if st.button("Analyze Sentiment") and user_input:
        with st.spinner("Analyzing sentiment..."):
            results = predict_sentiment([user_input], tokenizer, model)
            
            if results:
                result = results[0]
                display_sentiment_result(result, show_top_k)

def display_sentiment_result(result, show_top_k):
    """Display sentiment analysis result"""
    st.subheader("📊 Results")
    
    # Color code based on sentiment
    sentiment_color = {
        "positive": "🟢",
        "negative": "🔴", 
        "neutral": "🟡",
        "unknown": "⚪"
    }
    
    emoji = sentiment_color.get(result["predicted_label"].lower(), "⚪")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Predicted Sentiment",
            value=f"{emoji} {result['predicted_label'].upper()}"
        )
    
    with col2:
        st.metric(
            label="Confidence",
            value=f"{result['confidence']:.2%}"
        )
    
    # Confidence bar
    st.progress(float(result['confidence']))
    
    if show_top_k:
        st.subheader("Top Predictions")
        for label, prob in result["top_predictions"]:
            emoji = sentiment_color.get(label.lower(), "⚪")
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write(f"{emoji} **{label.upper()}**")
            with col2:
                st.write(f"{prob:.2%}")

def batch_analysis(tokenizer, model):
    st.header("📊 Batch Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a CSV file with a 'text' column",
        type=['csv'],
        help="Your CSV should have a column named 'text' containing the reviews to analyze"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if 'text' not in df.columns:
                st.error("CSV file must contain a 'text' column")
                return
            
            if st.button("Analyze All Reviews"):
                analyze_batch_data(df, tokenizer, model)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def analyze_batch_data(df, tokenizer, model):
    """Analyze batch data and display results"""
    with st.spinner("Analyzing batch data..."):
        # Analyze in batches to avoid memory issues
        batch_size = 16
        all_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch_texts = df['text'].iloc[i:i+batch_size].fillna('').astype(str).tolist()
            batch_results = predict_sentiment(batch_texts, tokenizer, model)
            all_results.extend(batch_results)
            
            # Update progress
            progress = min((i + batch_size) / len(df), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processed {min(i + batch_size, len(df))} / {len(df)} reviews")
        
        if all_results:
            display_batch_results(df, all_results)

def display_batch_results(df, all_results):
    """Display batch analysis results"""
    # Add results to dataframe
    results_df = df.copy()
    results_df['predicted_sentiment'] = [r['predicted_label'] for r in all_results]
    results_df['confidence'] = [r['confidence'] for r in all_results]
    
    # Display results
    st.subheader("Analysis Results")
    st.dataframe(results_df[['text', 'predicted_sentiment', 'confidence']].head(10))
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    sentiment_counts = results_df['predicted_sentiment'].value_counts()
    avg_confidence = results_df['confidence'].mean()
    
    with col1:
        st.metric("Total Reviews", len(results_df))
    with col2:
        st.metric("Positive", sentiment_counts.get('positive', 0))
    with col3:
        st.metric("Negative", sentiment_counts.get('negative', 0))
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    # Visualization
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Prepare data for plotting
    labels = list(sentiment_counts.index)
    counts = list(sentiment_counts.values)
    
    # Choose colors based on sentiment
    colors = []
    for label in labels:
        if 'positive' in label.lower():
            colors.append('green')
        elif 'negative' in label.lower():
            colors.append('red')
        else:
            colors.append('gray')
    
    bars = ax.bar(labels, counts, color=colors, alpha=0.7)
    ax.set_title('Sentiment Distribution')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv"
    )

def model_info(tokenizer, model):
    st.header("🤖 Model Information")
    
    # Model configuration
    if hasattr(model.config, 'model_type'):
        st.subheader("Model Configuration")
        config_dict = {
            'Model Type': model.config.model_type,
            'Number of Labels': model.config.num_labels,
            'Hidden Size': getattr(model.config, 'hidden_size', 'N/A'),
            'Number of Layers': getattr(model.config, 'num_hidden_layers', 'N/A')
        }
        
        for key, value in config_dict.items():
            st.write(f"**{key}:** {value}")
    
    # File check
    st.subheader("📁 File Check")
    model_path = "./bert-feedback-best"
    files_status = {}
    
    files_to_check = {
        'pytorch_model.bin': 'Model weights',
        'config.json': 'Model configuration',
        'vocab.txt': 'Tokenizer vocabulary',
        'tokenizer_config.json': 'Tokenizer configuration',
        'special_tokens_map.json': 'Special tokens mapping'
    }
    
    for file, description in files_to_check.items():
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            files_status[file] = f"✅ Found ({file_size:.1f} MB) - {description}"
        else:
            files_status[file] = f"❌ Missing - {description}"
    
    for file, status in files_status.items():
        st.write(status)
    
    # Example predictions
    st.subheader("🚀 Test the Model")
    example_texts = [
        "The product is excellent and delivery was super fast!",
        "The item was okay, nothing special.",
        "Terrible quality and poor customer service."
    ]
    
    for text in example_texts:
        if st.button(f"Test: '{text}'", key=text):
            with st.spinner("Predicting..."):
                results = predict_sentiment([text], tokenizer, model)
                if results:
                    result = results[0]
                    st.write(f"**Prediction:** {result['predicted_label']}")
                    st.write(f"**Confidence:** {result['confidence']:.2%}")
                    st.write("---")

if __name__ == "__main__":
    main()
