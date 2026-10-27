"""
Fine-tuning script for Sentence-BERT on educational & tutoring feedback data.

This script fine-tunes a sentence-transformers embedding model (e.g. all-MiniLM-L6-v2)
to specialize in:
1. Pedagogical concept equivalence (functions, variables, parameters).
2. Question-explanation alignment.
3. Feedback paraphrase and plagiarism detection.
4. Distinguishing compliant educational dialogues from off-platform/irrelevant text.

Usage:
    python scripts/train_sentence_bert.py --epochs 3 --batch_size 16 --output_dir models/fine_tuned_sentence_bert
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_dependencies():
    """Verify torch and sentence-transformers are installed."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import sentence_transformers
    except ImportError:
        missing.append("sentence-transformers")

    if missing:
        print(f"\n[ERROR] Missing required packages for fine-tuning: {', '.join(missing)}")
        print("Please install them with:")
        print(f"    pip install {' '.join(missing)}\n")
        sys.exit(1)


def load_dataset(data_path: str):
    """Load sentence pair training data."""
    if not os.path.exists(data_path):
        print(f"[WARNING] Data file '{data_path}' not found. Using default dataset.")
        data_path = Path(__file__).parent / "training_data.json"

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Loaded {len(data)} training pairs from {data_path}")
    return data


def train(
    base_model: str = "all-MiniLM-L6-v2",
    data_path: str = "scripts/training_data.json",
    output_dir: str = "models/fine_tuned_sentence_bert",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_steps: int = 10
):
    """Fine-tune the Sentence-BERT model."""
    check_dependencies()
    from sentence_transformers import SentenceTransformer, InputExample, losses, util
    from torch.utils.data import DataLoader
    import torch

    print("=" * 70)
    print("SENTENCE-BERT FINE-TUNING PIPELINE")
    print("=" * 70)
    print(f"Base Model:       {base_model}")
    print(f"Output Directory: {output_dir}")
    print(f"Epochs:           {epochs}")
    print(f"Batch Size:       {batch_size}")
    print(f"Learning Rate:    {lr}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Device:  {device.upper()}")
    print("=" * 70)

    # 1. Load Base Model
    print(f"\n[1/5] Loading base model '{base_model}'...")
    model = SentenceTransformer(base_model, device=device)

    # 2. Prepare Data
    print(f"[2/5] Preparing training pairs...")
    raw_pairs = load_dataset(data_path)

    train_examples = []
    for item in raw_pairs:
        train_examples.append(
            InputExample(
                texts=[item["sentence1"], item["sentence2"]],
                label=float(item["score"])
            )
        )

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # 3. Define Loss Function
    print("[3/5] Setting up CosineSimilarityLoss...")
    train_loss = losses.CosineSimilarityLoss(model=model)

    # 4. Fine-Tune
    print(f"[4/5] Training for {epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': lr},
        show_progress_bar=True
    )

    # 5. Save Model
    print(f"\n[5/5] Saving fine-tuned model to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    print(f"Model successfully saved to {output_dir}")

    # Evaluate on a test pair
    test_q = "How do we declare a function in Python?"
    test_ans = "Functions are declared using the def statement followed by arguments."
    test_irr = "Please wire money to my personal bank account."

    emb_q = model.encode(test_q, convert_to_tensor=True)
    emb_ans = model.encode(test_ans, convert_to_tensor=True)
    emb_irr = model.encode(test_irr, convert_to_tensor=True)

    sim_ans = util.cos_sim(emb_q, emb_ans).item()
    sim_irr = util.cos_sim(emb_q, emb_irr).item()

    print("\n" + "=" * 70)
    print("VERIFICATION OF FINE-TUNED EMBEDDINGS")
    print("=" * 70)
    print(f"Query:              '{test_q}'")
    print(f"Related Target:     '{test_ans}' -> Cosine Sim: {sim_ans:.4f} (High)")
    print(f"Irrelevant Target:  '{test_irr}' -> Cosine Sim: {sim_irr:.4f} (Low)")
    print("=" * 70)
    print("\nTo use this fine-tuned model in the QA Agent system:")
    print("Update config/config.yaml or set environment variable:")
    print(f"    export MODELS__SENTENCE_TRANSFORMER=\"{output_dir}\"")
    print("Or in config/config.yaml:")
    print(f"    models:\n      sentence_transformer: \"{output_dir}\"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Sentence-BERT for QA Agent System")
    parser.add_argument("--base_model", type=str, default="all-MiniLM-L6-v2", help="HuggingFace model name")
    parser.add_argument("--data_file", type=str, default="scripts/training_data.json", help="Path to training data JSON")
    parser.add_argument("--output_dir", type=str, default="models/fine_tuned_sentence_bert", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    train(
        base_model=args.base_model,
        data_path=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
