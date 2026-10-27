"""
CLI and Modular Training Script for Character-level LSTM on Text / Music Corpus.
Saves model weights (.pth) and tokenizer vocabulary (.json).
"""

import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.lstm_model import TextLSTM, generate_sequence
from utils.data_loader import CharTokenizer, get_data_loader

def train_model(
    corpus_text,
    model_save_path="models/checkpoints/lstm_model.pth",
    vocab_save_path="models/checkpoints/lstm_vocab.json",
    epochs=10,
    seq_len=64,
    batch_size=64,
    lr=0.003,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.2,
    step=3,
    device=None,
    progress_callback=None
):
    """
    Trains a TextLSTM on the provided text string or corpus.
    Supports real-time progress callbacks for Streamlit UI integration.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- Training LSTM Model ---")
    print(f"Device: {device}")
    print(f"Corpus size: {len(corpus_text)} characters")

    # 1. Tokenizer
    tokenizer = CharTokenizer(corpus_text)
    print(f"Vocabulary size: {tokenizer.vocab_size} unique characters")
    
    # Save vocabulary
    os.makedirs(os.path.dirname(vocab_save_path), exist_ok=True)
    tokenizer.save(vocab_save_path)
    print(f"Vocabulary saved to {vocab_save_path}")

    # 2. Encode corpus
    encoded_data = tokenizer.encode(corpus_text)
    
    # 3. DataLoader
    loader = get_data_loader(
        encoded_data, 
        seq_len=seq_len, 
        batch_size=batch_size, 
        step=step, 
        shuffle=True
    )
    total_batches = len(loader)
    print(f"Total training batches per epoch: {total_batches}")

    # 4. Model initialization
    model = TextLSTM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)

    history = {"epoch": [], "loss": [], "perplexity": []}
    seed_prompt = corpus_text[:20].strip() or " "

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for b_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            
            # Reshape for CrossEntropyLoss: (batch_size * seq_len, vocab_size) vs (batch_size * seq_len)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), targets.view(-1))
            loss.backward()

            # Gradient clipping to prevent exploding gradients in LSTM
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if progress_callback and b_idx % max(1, total_batches // 5) == 0:
                progress_callback({
                    "stage": "batch",
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "batch": b_idx + 1,
                    "total_batches": total_batches,
                    "batch_loss": loss.item()
                })

        avg_loss = total_loss / max(1, batch_count)
        perplexity = float(torch.exp(torch.tensor(avg_loss)).item())
        scheduler.step()

        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["perplexity"].append(perplexity)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

        # Generate sample text every few epochs to verify learning
        if epoch % max(1, epochs // 3) == 0 or epoch == epochs:
            sample_preview = generate_sequence(
                model=model,
                char2idx=tokenizer.char2idx,
                idx2char=tokenizer.idx2char,
                seed_text=seed_prompt,
                max_length=60,
                temperature=0.8,
                device=device
            )
            print(f"Sample preview (temp=0.8):\n{sample_preview.strip()}\n" + "-"*40)

        if progress_callback:
            progress_callback({
                "stage": "epoch",
                "epoch": epoch,
                "total_epochs": epochs,
                "loss": avg_loss,
                "perplexity": perplexity,
                "history": history
            })

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")

    # 5. Save model checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "history": history
    }
    torch.save(checkpoint, model_save_path)
    print(f"Model saved to {model_save_path}")

    return model, tokenizer, history


def main():
    parser = argparse.ArgumentParser(description="Train LSTM on Text / Music Corpus")
    parser.add_argument("--data_file", type=str, default="data/poetry_shakespeare.txt", help="Path to input text corpus")
    parser.add_argument("--model_name", type=str, default="poetry", help="Prefix name for saved model")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="LSTM hidden dimension")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--max_chars", type=int, default=120000, help="Max characters to use from corpus for quick training")
    
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        print(f"Error: Data file {args.data_file} not found. Running prepare_data.py first.")
        import prepare_data
        prepare_data.main()

    with open(args.data_file, "r", encoding="utf-8") as f:
        corpus = f.read()

    # Limit corpus length if requested for fast CPU training
    if args.max_chars and len(corpus) > args.max_chars:
        print(f"Subsetting corpus to {args.max_chars} chars for rapid convergence.")
        corpus = corpus[:args.max_chars]

    model_path = f"models/checkpoints/{args.model_name}_model.pth"
    vocab_path = f"models/checkpoints/{args.model_name}_vocab.json"

    train_model(
        corpus_text=corpus,
        model_save_path=model_path,
        vocab_save_path=vocab_path,
        epochs=args.epochs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        step=3
    )

if __name__ == "__main__":
    main()
