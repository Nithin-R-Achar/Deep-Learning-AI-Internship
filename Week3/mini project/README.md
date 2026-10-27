# 🎼 MuseLSTM: Creative Text & Music Generation System

An end-to-end deep learning system powered by PyTorch multi-layer Long Short-Term Memory (LSTM) networks for creative text, code, and symbolic music generation with real-time temperature control and audio synthesis.

---

## 🌟 Key Features

1. **Multi-Domain Creative Generation**:
   - 🎭 **Classic Poetry & Drama**: Shakespearean sonnets, romantic meter, and theatrical dialogues.
   - 🎵 **ABC Folk Music & Melodies**: Symbolic ASCII music notation converted into playable audio (WAV) via an in-memory harmonic waveform synthesizer.
   - 🎤 **Song Lyrics**: Verse-chorus pop/rock lyrical rhyming patterns and hooks.
   - 💻 **Python Code**: Algorithmic definitions, function signatures, loops, and control flow.
   - 📁 **Custom Text Dataset**: Train your own custom model in the browser on any `.txt` file!

2. **Interactive Streamlit Web Application**:
   - **Temperature Control ($T \\in [0.1, 2.0]$)**:
     - $T < 0.4$: Conservative & deterministic patterns.
     - $0.7 \\le T \\le 0.85$: Golden balance of linguistic fluency and novelty.
     - $T > 1.2$: Wild, highly creative, unexpected linguistic exploration.
   - **Max Length Slider**: Set autoregressive sequence length from 50 to 1,200 tokens.
   - **Top-K Sampling Constraint**: Filter out low-probability tail tokens to eliminate garbled text.
   - **Interactive Audio Synthesizer**: Directly listen to and download generated ABC tunes as 16-bit PCM WAV audio.
   - **Interactive Training Studio**: Live training in the browser with real-time loss tracking and metric cards.

3. **Core Model Architecture**:
   - Character-level vocabulary embedding (128-dim).
   - Multi-layer LSTM recurrent layers (256-dim hidden units) with inter-layer Dropout.
   - Linear projection decoder to vocabulary logits.
   - Gradient clipping ($\\|\mathbf{g}\\| \\le 5.0$) and adaptive StepLR learning rate scheduling.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Training Corpora
```bash
python prepare_data.py
```

### 3. Train from Command Line (Optional)
```bash
# Train Poetry model
python train.py --data_file data/poetry_shakespeare.txt --model_name poetry --epochs 5

# Train Music model
python train.py --data_file data/abc_music.txt --model_name music --epochs 4
```

### 4. Launch the Streamlit Web App
```bash
streamlit run app.py
```

---

## 📊 Temperature-Scaled Sampling Mechanics

The model outputs unnormalized logits $z_k$ for each character $k$ in the vocabulary. The next character is sampled according to the temperature-adjusted softmax distribution:

$$P(x_{t+1} = k) = \frac{\exp(z_k / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}$$

- When $T \to 0$, the probability distribution approaches a one-hot Dirac distribution on the maximum logit (greedy decoding).
- When $T = 1.0$, sampling reflects the empirical conditional distribution learned during training.
- When $T > 1.0$, the distribution flattens, increasing the entropy and diversity of sampled sequences.
