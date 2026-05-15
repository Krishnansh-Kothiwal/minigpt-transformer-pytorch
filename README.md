# MiniGPT: Character-Level Transformer Language Model

A decoder-only Transformer language model built from scratch in PyTorch. Trained on character-level Shakespeare text, this project implements the core GPT architecture — including causal self-attention, multi-head attention, residual connections, and autoregressive generation — in ~200 lines of clean, readable code.

## Disclaimer

This project is an educational implementation inspired by Andrej Karpathy’s nanoGPT/ng-video-lecture workflow and was built to better understand transformer internals and autoregressive language modeling.

## Features

- **Decoder-only Transformer** — Full GPT-style architecture with configurable depth and width
- **Causal Self-Attention** — Masked attention preventing future token leakage
- **Multi-Head Attention** — Parallel attention heads with learned projections
- **Positional Embeddings** — Learned position encodings for sequence order
- **Residual Connections** — Skip connections around attention and feedforward blocks
- **Layer Normalization** — Pre-norm architecture for stable training (Pre-LN Transformer)
- **Feedforward Network** — Two-layer MLP with 4× expansion ratio and ReLU activation
- **GPT-style Weight Initialization** — Normal initialization with std=0.02
- **Autoregressive Generation** — Token-by-token multinomial sampling from model probabilities
- **Bigram Baseline** — Minimal baseline model for comparison

## Architecture

```
Input Token IDs
      │
      ▼
┌─────────────────┐
│ Token Embedding  │  (vocab_size → n_embd)
│       +          │
│ Position Embed   │  (block_size → n_embd)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transformer Block│ ×N
│                  │
│  ┌────────────┐  │
│  │ LayerNorm  │  │
│  │ Multi-Head │  │
│  │ Attention  │──┤── Residual Connection
│  └────────────┘  │
│  ┌────────────┐  │
│  │ LayerNorm  │  │
│  │ FeedForward│──┤── Residual Connection
│  └────────────┘  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final LayerNorm │
│  Linear → Logits │  (n_embd → vocab_size)
└─────────────────┘
         │
         ▼
   Next Token Prediction
```

### Model Configuration

| Hyperparameter  | Value  |
|-----------------|--------|
| Embedding dim   | 384    |
| Attention heads | 6      |
| Layers          | 6      |
| Context length  | 256    |
| Dropout         | 0.2    |
| Parameters      | ~10.7M |

## Technical Concepts Demonstrated

- **Transformer architecture** — Attention Is All You Need (Vaswani et al., 2017)
- **Causal masking** — Lower-triangular mask for autoregressive language modeling
- **Scaled dot-product attention** — `Q·Kᵀ / √d_k` for stable gradients
- **Pre-LayerNorm** — Normalization before sublayers (vs. Post-LN in original paper)
- **Cross-entropy loss** on next-token prediction
- **AdamW optimizer** with tuned learning rate
- **Train / validation split** with periodic loss evaluation
- **Character-level tokenization** — No external tokenizer dependency

## Skills Demonstrated

`Python` · `PyTorch` · `Deep Learning` · `NLP` · `Transformer Architecture` · `Attention Mechanisms` · `Language Modeling` · `Neural Network Design` · `GPU Training`

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/minigpt-transformer-pytorch.git
cd minigpt-transformer-pytorch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+ (CUDA optional, falls back to CPU)

## Usage

### Train the full Transformer model

```bash
python train.py
```

Or run the original all-in-one script:

```bash
python gpt.py
```

### Generate text from a trained checkpoint

```bash
python generate.py --checkpoint checkpoint.pt --length 1000
```

### Train the bigram baseline (for comparison)

```bash
python bigram.py
```

## Sample Output

After training on Tiny Shakespeare:

```
DUKE VINCENTIO:
Well have been bolly poor late
Is the lords.

ABELLA:
Let's found: I will kind him;
I do braw'sy him business wherein far his face.

LUCENTIO:
He is last afford: make him diseably to London,
Take him great Hastings, boldness in his natic keeps,
To oftragn lost me ready glust through the house.
Why chose that I dares it be a Montague.
```

> The model learns character-level patterns including dialogue structure, character names, iambic-ish rhythm, and Shakespearean vocabulary — all from a ~10M parameter model.

## Project Structure

```
minigpt/
├── gpt.py            # Full Transformer implementation + training
├── bigram.py          # Bigram baseline model
├── train.py           # Training script with checkpoint saving
├── generate.py        # Text generation from saved checkpoints
├── input.txt          # Tiny Shakespeare training corpus (~1MB)
├── requirements.txt   # Python dependencies
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Limitations

- **Character-level only** — No subword tokenization (BPE/SentencePiece)
- **Small corpus** — Trained on ~1MB of Shakespeare; not generalizable
- **No learning rate scheduling** — Fixed LR throughout training
- **Single-GPU** — No distributed training support
- **No beam search** — Generation uses simple multinomial sampling

## Future Improvements

- [ ] Add BPE tokenization for better text quality
- [ ] Implement learning rate warmup + cosine decay
- [ ] Add top-k / top-p (nucleus) sampling
- [ ] Support training on custom text datasets
- [ ] Add mixed-precision training (FP16/BF16)
- [ ] Implement KV-cache for faster inference

## Resume Bullet

> Built a decoder-only Transformer language model from scratch in PyTorch with causal self-attention, multi-head attention, residual blocks, layer normalization, and autoregressive text generation.

## Acknowledgments

Architecture inspired by [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) and Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

## License

MIT
