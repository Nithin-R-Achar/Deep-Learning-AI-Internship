"""
Streamlit Web Application for LSTM Creative Text & Music Generation.
Allows users to set temperature, max length, domain, seed prompt,
listen to synthesized ABC music, and train custom LSTM models interactively.
"""

import os
import sys
import io
import time
import json
import torch
import streamlit as st
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.lstm_model import TextLSTM, generate_sequence
from utils.data_loader import CharTokenizer
from utils.music_synth import abc_to_wav_bytes

# --- Page Configuration ---
st.set_page_config(
    page_title="MuseLSTM: Creative Text & Music Generation",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Glassmorphism & Sleek Dark Aesthetics) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Background and containers */
    .main {
        background: radial-gradient(circle at 10% 20%, rgba(20, 24, 40, 1) 0%, rgba(10, 12, 22, 1) 90%);
        color: #e2e8f0;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .glass-card-accent {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }

    /* Gradient header text */
    .gradient-header {
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Pill badges */
    .pill-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
    }

    /* Stat metrics */
    .stat-box {
        text-align: center;
        padding: 16px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.35);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(124, 58, 237, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- Helper: Model & Tokenizer Loader ---
@st.cache_resource
def load_trained_model(model_path, vocab_path):
    """Loads model weights and tokenizer from checkpoint files."""
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        return None, None, f"Checkpoint not found at {model_path}"

    try:
        tokenizer = CharTokenizer.load(vocab_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        vocab_size = checkpoint.get("vocab_size", tokenizer.vocab_size)
        embedding_dim = checkpoint.get("embedding_dim", 128)
        hidden_dim = checkpoint.get("hidden_dim", 256)
        num_layers = checkpoint.get("num_layers", 2)
        dropout = checkpoint.get("dropout", 0.2)

        model = TextLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)


# --- Available Domains and Default Configs ---
DOMAINS = {
    "poetry": {
        "title": "Shakespeare & Classic Poetry",
        "icon": "🎭",
        "data_file": "data/poetry_shakespeare.txt",
        "model_file": "models/checkpoints/poetry_model.pth",
        "vocab_file": "models/checkpoints/poetry_vocab.json",
        "sample_prompts": [
            "Shall I compare thee to a summer's day?",
            "When in eternal lines to time thou grow'st,",
            "O never say that I was false of heart,",
            "The expense of spirit in a waste of shame"
        ],
        "default_temp": 0.75,
        "description": "Iambic rhythm, romantic sonnets, and dramatic dialogue."
    },
    "music": {
        "title": "ABC Folk Music & Melodies",
        "icon": "🎵",
        "data_file": "data/abc_music.txt",
        "model_file": "models/checkpoints/music_model.pth",
        "vocab_file": "models/checkpoints/music_vocab.json",
        "sample_prompts": [
            "X: 1\nT: Novel Reel\nM: 4/4\nL: 1/8\nK: Dmaj\n|:FAAF AFED|",
            "X: 2\nT: Celtic Breeze\nM: 4/4\nL: 1/8\nK: Gmaj\n|:G2BG dGBG|",
            "X: 3\nT: Starlight Jig\nM: 6/8\nL: 1/8\nK: Edor\n|:E2B B2A|",
            "|:faab afdf|a2fd edBd|"
        ],
        "default_temp": 0.8,
        "description": "Symbolic ASCII ABC musical notation with playable audio synthesis!"
    },
    "lyrics": {
        "title": "Song Lyrics & Chorus",
        "icon": "🎤",
        "data_file": "data/song_lyrics.txt",
        "model_file": "models/checkpoints/lyrics_model.pth",
        "vocab_file": "models/checkpoints/lyrics_vocab.json",
        "sample_prompts": [
            "[Verse 1]\nWalking through the city in the neon light",
            "[Chorus]\nOh, take me higher where the river flows",
            "[Bridge]\nCan you hear the bassline groove?",
            "Tonight we are the fire, tonight we are the light"
        ],
        "default_temp": 0.85,
        "description": "Pop, rock, and rhythmic verses with emotional hooks and rhyming."
    },
    "code": {
        "title": "Python & Algorithmic Code",
        "icon": "💻",
        "data_file": "data/python_code.txt",
        "model_file": "models/checkpoints/code_model.pth",
        "vocab_file": "models/checkpoints/code_vocab.json",
        "sample_prompts": [
            "def binary_search(elements, query):",
            "class DoublyLinkedList:\n    def __init__(self):",
            "def quick_sort(arr):",
            "def fibonacci(n):"
        ],
        "default_temp": 0.65,
        "description": "Python structures, syntax, function definitions, and logic."
    }
}

# --- Sidebar Controls ---
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h2 style="margin: 0; font-size: 1.6rem; color: #a5b4fc;">🎼 MuseLSTM Studio</h2>
    <p style="font-size: 0.85rem; color: #94a3b8; margin: 4px 0 0 0;">Deep Generative Sequence Modeling</p>
</div>
""", unsafe_allow_html=True)

domain_key = st.sidebar.selectbox(
    "Select Creative Domain",
    options=list(DOMAINS.keys()),
    format_func=lambda k: f"{DOMAINS[k]['icon']} {DOMAINS[k]['title']}"
)
domain_info = DOMAINS[domain_key]

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Sampling Parameters")

# Temperature Slider
temp_col1, temp_col2 = st.sidebar.columns([3, 1])
temperature = st.sidebar.slider(
    "Creativity Temperature",
    min_value=0.1,
    max_value=2.0,
    value=domain_info["default_temp"],
    step=0.05,
    help="Higher values (e.g. 1.2+) produce bolder, more unpredictable tokens. Lower values (e.g. 0.3) produce safer, repetitive structures."
)

# Temperature explanation badge
if temperature < 0.4:
    temp_badge = "🔒 Conservative & Predictable"
    temp_color = "#38bdf8"
elif temperature < 0.85:
    temp_badge = "🎯 Balanced & Coherent (Sweet Spot)"
    temp_color = "#34d399"
elif temperature < 1.3:
    temp_badge = "🎨 Creative & Adventurous"
    temp_color = "#f59e0b"
else:
    temp_badge = "⚡ Wild & Chaotic"
    temp_color = "#f43f5e"

st.sidebar.markdown(f"<span style='color: {temp_color}; font-size: 0.85rem; font-weight: 600;'>{temp_badge}</span>", unsafe_allow_html=True)

# Max Length Slider
max_length = st.sidebar.slider(
    "Maximum Length (Characters)",
    min_value=50,
    max_value=1200,
    value=400,
    step=50,
    help="Total number of characters to generate autoregressively."
)

# Top-K constraint
top_k_enabled = st.sidebar.checkbox("Enable Top-K Sampling Filter", value=True)
if top_k_enabled:
    top_k = st.sidebar.slider("Top-K Value", min_value=5, max_value=80, value=35, step=5)
else:
    top_k = None

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.8rem; color: #64748b; line-height: 1.4;">
    <strong>Architecture Specs:</strong><br>
    • Recurrent Unit: Multi-Layer LSTM<br>
    • Cell States: Hidden <code>h_t</code> & Cell <code>c_t</code><br>
    • Tokenizer: Character-level encoding<br>
    • Audio Engine: SciPy 22kHz Waveform Synthesizer
</div>
""", unsafe_allow_html=True)


# --- Main Navigation Tabs ---
tab_gen, tab_train, tab_theory = st.tabs([
    "✨ Creative Generator",
    "🏋️ Model Training Studio",
    "🧠 LSTM Architecture & Theory"
])


# ==============================================================================
# TAB 1: CREATIVE GENERATOR
# ==============================================================================
with tab_gen:
    # Header Banner
    st.markdown(f"""
    <div class="glass-card-accent">
        <h1 class="gradient-header" style="margin: 0; font-size: 2.2rem;">
            {domain_info['icon']} {domain_info['title']} Generator
        </h1>
        <p style="color: #cbd5e1; margin-top: 8px; font-size: 1.05rem;">
            {domain_info['description']}
        </p>
        <div>
            <span class="pill-badge">Temperature: {temperature:.2f}</span>
            <span class="pill-badge">Max Length: {max_length} chars</span>
            <span class="pill-badge">Top-K: {top_k if top_k else 'Off'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check model availability
    model, tokenizer, error_msg = load_trained_model(
        os.path.join(BASE_DIR, domain_info["model_file"]),
        os.path.join(BASE_DIR, domain_info["vocab_file"])
    )

    if model is None:
        st.warning(f"⚠️ Model checkpoint for **{domain_info['title']}** is not yet trained or found.")
        st.info("You can train this domain instantly using the button below or visit the **🏋️ Model Training Studio** tab.")
        
        train_quick_col1, _ = st.columns([1, 2])
        with train_quick_col1:
            if st.button(f"⚡ Train Quick {domain_info['title']} Model Now", key="quick_train_btn"):
                with st.spinner("Training LSTM model on CPU (takes ~30-60 seconds)..."):
                    from train import train_model
                    data_file_path = os.path.join(BASE_DIR, domain_info["data_file"])
                    if not os.path.exists(data_file_path):
                        import prepare_data
                        prepare_data.main()
                    with open(data_file_path, "r", encoding="utf-8") as f:
                        text_corpus = f.read()[:30000] # Subsampled for rapid CPU completion
                    
                    train_model(
                        corpus_text=text_corpus,
                        model_save_path=os.path.join(BASE_DIR, domain_info["model_file"]),
                        vocab_save_path=os.path.join(BASE_DIR, domain_info["vocab_file"]),
                        epochs=4,
                        seq_len=64,
                        batch_size=64,
                        lr=0.003,
                        hidden_dim=256,
                        embedding_dim=128
                    )
                    st.cache_resource.clear()
                    st.success("Model trained successfully! Refreshing...")
                    st.rerun()

    else:
        # Prompt Selection & Input
        col_prompt, col_preset = st.columns([3, 2])
        
        with col_preset:
            selected_sample = st.selectbox(
                "💡 Quick Sample Prompts",
                options=["(Custom Prompt)"] + domain_info["sample_prompts"]
            )

        with col_prompt:
            default_prompt_val = domain_info["sample_prompts"][0] if selected_sample == "(Custom Prompt)" else selected_sample
            seed_input = st.text_area(
                "Enter Seed Text / Prompt:",
                value=default_prompt_val,
                height=100,
                help="The initial text provided to prime the LSTM's internal hidden states."
            )

        # Generate Trigger
        btn_col, clear_col = st.columns([2, 5])
        with btn_col:
            generate_clicked = st.button("🚀 Generate Novel Sequence", use_container_width=True)

        if generate_clicked:
            if not seed_input.strip():
                st.error("Please enter at least one character for the seed prompt.")
            else:
                progress_placeholder = st.empty()
                progress_placeholder.info("🔄 Priming recurrent cell states & sampling autoregressively...")
                
                start_gen_time = time.time()
                generated_output = generate_sequence(
                    model=model,
                    char2idx=tokenizer.char2idx,
                    idx2char=tokenizer.idx2char,
                    seed_text=seed_input,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    device=torch.device('cpu')
                )
                gen_duration = time.time() - start_gen_time
                progress_placeholder.empty()

                # Display Results
                st.markdown("### 📜 Generated Sequence")
                
                # If code, render formatted code block
                if domain_key == "code":
                    st.code(generated_output, language="python")
                else:
                    st.markdown(f"""
                    <div class="glass-card" style="font-size: 1.05rem; line-height: 1.7; white-space: pre-wrap; border-left: 4px solid #8b5cf6;">
{generated_output}
                    </div>
                    """, unsafe_allow_html=True)

                # Audio Playback for ABC Music
                if domain_key == "music" or ("X:" in generated_output and "K:" in generated_output):
                    st.markdown("---")
                    st.subheader("🎧 Synthesized Music Audio Playback")
                    
                    audio_col1, audio_col2 = st.columns([2, 1])
                    with audio_col1:
                        bpm_val = st.slider("Playback Tempo (BPM)", min_value=80, max_value=200, value=130, step=10)
                        try:
                            wav_bytes = abc_to_wav_bytes(generated_output, bpm=bpm_val)
                            st.audio(wav_bytes, format="audio/wav")
                            st.download_button(
                                label="⬇️ Download Synthesized Melody (.wav)",
                                data=wav_bytes,
                                file_name=f"lstm_melody_{int(time.time())}.wav",
                                mime="audio/wav"
                            )
                        except Exception as audio_err:
                            st.warning(f"Audio synthesizer note: {audio_err}")

                    with audio_col2:
                        st.markdown("""
                        <div class="glass-card" style="padding: 16px;">
                            <strong style="color: #38bdf8;">ABC Music Notation</strong><br>
                            <span style="font-size: 0.85rem; color: #94a3b8;">
                            The LSTM generated raw standard ABC notation, which our harmonic synthesizer parsed into musical notes, frequencies, and a 16-bit PCM waveform!
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                # Generation Metrics & Download
                st.markdown("---")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Generated Length", f"{len(generated_output)} chars")
                m_col2.metric("Generation Time", f"{gen_duration:.3f} s")
                m_col3.metric("Tokens/Second", f"{(len(generated_output)/max(0.01, gen_duration)):.1f}")
                m_col4.metric("Sampling Temp", f"{temperature:.2f}")

                # Download Text
                st.download_button(
                    label="💾 Download Generated Text (.txt)",
                    data=generated_output,
                    file_name=f"lstm_generated_{domain_key}_{int(time.time())}.txt",
                    mime="text/plain"
                )


# ==============================================================================
# TAB 2: MODEL TRAINING STUDIO
# ==============================================================================
with tab_train:
    st.markdown("""
    <div class="glass-card">
        <h2 class="gradient-header" style="margin: 0;">🏋️ Interactive LSTM Training Studio</h2>
        <p style="color: #94a3b8; margin-top: 6px;">
            Train a custom character-level LSTM on your own text dataset or fine-tune on standard corpora with real-time loss tracking.
        </p>
    </div>
    """, unsafe_allow_html=True)

    t_col1, t_col2 = st.columns([1, 1])

    with t_col1:
        st.subheader("1. Data Source")
        data_source = st.radio(
            "Select Training Corpus",
            options=["Pre-packaged Domain Corpus", "Upload Custom Text File (.txt)"]
        )

        training_text = ""
        dataset_name = "custom"

        if data_source == "Pre-packaged Domain Corpus":
            domain_choice = st.selectbox("Corpus Domain", options=list(DOMAINS.keys()), format_func=lambda k: DOMAINS[k]["title"])
            chosen_file = os.path.join(BASE_DIR, DOMAINS[domain_choice]["data_file"])
            dataset_name = domain_choice
            if os.path.exists(chosen_file):
                with open(chosen_file, "r", encoding="utf-8") as f:
                    training_text = f.read()
                st.success(f"Loaded {DOMAINS[domain_choice]['title']} ({len(training_text):,} characters)")
            else:
                st.error("Corpus file not found.")
        else:
            uploaded_file = st.file_uploader("Upload .txt file (e.g. lyrics, poems, recipes, books)", type=["txt"])
            if uploaded_file is not None:
                training_text = uploaded_file.read().decode("utf-8", errors="ignore")
                dataset_name = "user_custom"
                st.success(f"Uploaded custom file: {uploaded_file.name} ({len(training_text):,} characters)")

        # Subset slider
        if training_text:
            max_avail = min(len(training_text), 150000)
            chars_to_use = st.slider(
                "Corpus Subsample Size (Chars)",
                min_value=5000,
                max_value=max_avail,
                value=min(max_avail, 25000),
                step=5000,
                help="Subsetting the corpus allows fast training on CPU while preserving vocabulary diversity."
            )
            training_text = training_text[:chars_to_use]

    with t_col2:
        st.subheader("2. Hyperparameters")
        hyp_c1, hyp_c2 = st.columns(2)
        train_epochs = hyp_c1.slider("Epochs", min_value=1, max_value=15, value=4, step=1)
        train_seq_len = hyp_c2.slider("Sequence Length", min_value=32, max_value=96, value=64, step=16)
        train_hidden = hyp_c1.selectbox("LSTM Hidden Dim", options=[128, 256, 384], index=1)
        train_lr = hyp_c2.select_slider("Learning Rate", options=[0.001, 0.002, 0.003, 0.005], value=0.003)
        train_batch = 64

        st.markdown(f"""
        <div class="stat-box" style="margin-top: 15px;">
            <span class="stat-label">Estimated Training Batches</span>
            <div class="stat-number">{max(1, (len(training_text) - train_seq_len) // (3 * train_batch))}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    start_train_btn = st.button("🚀 Start Interactive LSTM Training", use_container_width=True)

    if start_train_btn:
        if not training_text or len(training_text) < 1000:
            st.error("Please provide or upload a text corpus with at least 1,000 characters.")
        else:
            from train import train_model
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            loss_chart_placeholder = st.empty()
            preview_placeholder = st.empty()

            target_model_path = os.path.join(BASE_DIR, f"models/checkpoints/{dataset_name}_model.pth")
            target_vocab_path = os.path.join(BASE_DIR, f"models/checkpoints/{dataset_name}_vocab.json")

            epoch_losses = []
            
            def on_progress(info):
                if info["stage"] == "batch":
                    overall_frac = ((info["epoch"] - 1) + (info["batch"] / info["total_batches"])) / info["total_epochs"]
                    progress_bar.progress(min(1.0, overall_frac))
                    status_text.markdown(f"**Epoch {info['epoch']}/{info['total_epochs']}** | Batch {info['batch']}/{info['total_batches']} | Current Batch Loss: `{info['batch_loss']:.4f}`")
                elif info["stage"] == "epoch":
                    epoch_losses.append(info["loss"])
                    loss_chart_placeholder.line_chart({"Training Cross-Entropy Loss": epoch_losses})

            st.toast("Training initiated!", icon="🔥")
            try:
                trained_model, tokenizer, history = train_model(
                    corpus_text=training_text,
                    model_save_path=target_model_path,
                    vocab_save_path=target_vocab_path,
                    epochs=train_epochs,
                    seq_len=train_seq_len,
                    batch_size=train_batch,
                    lr=train_lr,
                    hidden_dim=train_hidden,
                    embedding_dim=128,
                    progress_callback=on_progress
                )
                progress_bar.progress(1.0)
                status_text.success("🎉 Training complete! Checkpoint and vocabulary saved successfully.")
                st.cache_resource.clear()

                # Generate a victory sample
                preview_sample = generate_sequence(
                    model=trained_model,
                    char2idx=tokenizer.char2idx,
                    idx2char=tokenizer.idx2char,
                    seed_text=training_text[:20].strip(),
                    max_length=200,
                    temperature=0.8
                )
                preview_placeholder.markdown(f"""
                ### 🎯 Freshly Generated Sample:
                ```
{preview_sample}
                ```
                """)

            except Exception as train_exc:
                st.error(f"Training error: {train_exc}")


# ==============================================================================
# TAB 3: ARCHITECTURE & THEORY
# ==============================================================================
with tab_theory:
    st.markdown("""
    <div class="glass-card">
        <h2 class="gradient-header" style="margin: 0;">🧠 LSTM Sequence Generation Mechanics</h2>
        <p style="color: #cbd5e1; margin-top: 6px;">
            How Long Short-Term Memory recurrent networks learn sequential syntax and generate creative outputs with temperature-scaled sampling.
        </p>
    </div>
    """, unsafe_allow_html=True)

    th_col1, th_col2 = st.columns(2)

    with th_col1:
        st.markdown("""
        ### 1. The LSTM Cell & Gating Equations
        Standard Recurrent Neural Networks (RNNs) suffer from vanishing and exploding gradients when propagating errors over long time horizons.
        
        The **LSTM (Hochreiter & Schmidhuber)** overcomes this with a conveyor-belt **Cell State** ($c_t$) regulated by three non-linear gates:

        1. **Forget Gate ($f_t$):** Decides what information to discard from the past:
        $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

        2. **Input Gate ($i_t$ & $\\tilde{c}_t$):** Decides which new candidates to store:
        $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
        $$\\tilde{c}_t = \\tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

        3. **Cell State Update ($c_t$):** Linear constant error carousel:
        $$c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t$$

        4. **Output Gate ($o_t$) & Hidden State ($h_t$):**
        $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
        $$h_t = o_t \\odot \\tanh(c_t)$$
        """)

    with th_col2:
        st.markdown("""
        ### 2. Temperature-Scaled Stochastic Sampling
        At each generation step $t$, the model outputs raw unnormalized logits $z \\in \\mathbb{R}^{|V|}$ for each character in the vocabulary $V$.
        
        To balance predictability and creativity, logits are scaled by temperature $T > 0$:

        $$P(x_{t+1} = k) = \\frac{\\exp(z_k / T)}{\\sum_{j=1}^{|V|} \\exp(z_j / T)}$$

        #### Mathematical Effect of Temperature $T$:
        - **$T \\to 0$ (Greedy / Argmax):** The highest logit dominates; probability collapses to 1.0 on the most frequent token. Output is repetitive and deterministic.
        - **$T = 1.0$ (Standard Softmax):** Matches the true empirical categorical distribution learned during training.
        - **$T > 1.0$ (High Entropy / Creative):** Flattens the distribution towards a uniform spread. Rare and unexpected tokens have a much higher chance of being chosen, producing avant-garde combinations or phonetic surprises!
        - **Top-K Filtering:** Restricts candidate selection to the top $K$ most probable characters before re-normalizing softmax, preventing wild out-of-distribution hallucinations.
        """)

    st.markdown("---")
    st.markdown("""
    ### 3. Musical Representation: ABC Symbolic Notation
    Unlike raw audio waveforms (which require 44,100 samples per second), **ABC notation** represents music symbolically as plain ASCII text.
    For example:
    ```
    M: 4/4          <- Meter (time signature)
    L: 1/8          <- Default note length
    K: Dmaj         <- Key signature (D major)
    |:FAAF AFED|FAAF ABde|fBBA Bcde|f2af efdB:|
    ```
    An LSTM trained on ABC notation learns:
    1. Structural headers (`M:`, `L:`, `K:`)
    2. Bar length constraints (e.g. 8 eighth notes per bar)
    3. Harmonic progressions within key signatures
    4. Repetition patterns and musical cadences (`|: ... :|`)
    """)
