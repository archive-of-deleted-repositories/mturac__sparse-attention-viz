# 🔥 Sparse Attention Visualizer: MiniMax-Style on CPU

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![No GPU Required](https://img.shields.io/badge/GPU-Not%20Required-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Languages:** English &nbsp;|&nbsp; [Türkçe](README.tr.md) &nbsp;|&nbsp; [Français](README.fr.md) &nbsp;|&nbsp; [Deutsch](README.de.md) &nbsp;|&nbsp; [中文](README.zh.md)

---

![Sparse Attention Demo](sparse-attention-demo.svg)

**No GPU? No problem. Visualize and understand the sparse attention technology used by MiniMax on 1M tokens — running entirely on CPU.** 🧠✨

---

## 🎯 Why This Project?

Large language models slow down dramatically and consume excessive memory when processing long sequences, due to the **O(n²) attention mechanism**.

Companies like **MiniMax** solve this with **sparse attention**:
- Instead of attending to every token, look only at the important ones
- Achieve **9.7× speedup** at 1M tokens
- Cut memory usage significantly

**Yet nobody has visualized how it actually works.** 🤔

This project:
- ✅ **Implements Top-k sparse attention** (MiniMax-style) from scratch
- ✅ **No GPU required** — runs on CPU
- ✅ **Visualizes** which words are attended to, in color
- ✅ **Educational** — for students, researchers, and the curious
- ✅ **Open source** — modify and extend freely

---

## 🌟 Features

### 🔬 3 Attention Mechanisms
1. **Full Attention** — Standard O(n²) baseline
2. **Random Sparse** — Randomly sampled tokens
3. **Top-k Sparse** — Attends only to the top-k scoring tokens (MiniMax-style) ⭐

### 📊 Rich Visualizations
- **Attention Heatmaps** — Who attends to whom?
- **Token Selection Maps** — Which words were selected vs. skipped?
- **Speed–Accuracy Tradeoff** — How much speedup, how much quality loss?
- **Interactive Plots** — Explore with Plotly

### ⚡ Performance Metrics
- Inference time (ms)
- Memory usage (MB)
- Accuracy / F1 score
- Speedup ratio

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/mturac/sparse-attention-viz.git
cd sparse-attention-viz

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run!
python main.py
```

**No GPU? No problem.** Everything runs on CPU. 💪

---

## 🚀 Quick Start

### Demo Notebook
```bash
jupyter notebook demo.ipynb
```

### Run All Experiments
```bash
python main.py --dataset imdb --samples 1000 --k-values 5 10 20 50 100
```

### Visualization Only
```bash
python attention_viz.py --example "I love this movie, it's amazing!"
```

---

## 📈 Example Results

| Full Attention | Random Sparse | Top-k Sparse |
|:-:|:-:|:-:|
| Attends to all tokens | Random selection | Only top-k tokens |
| O(n²) | O(n·k) | O(n·k) |
| Baseline | Weak baseline | **MiniMax-style** ⭐ |

---

## 🔬 Experiments

### Experiment 1: Varying k
```bash
python main.py --k-values 5 10 20 50 100 --samples 500
```

### Experiment 2: Dataset Comparison
```bash
python main.py --dataset imdb --samples 1000
python main.py --dataset agnews --samples 1000
```

### Experiment 3: Sequence Length Analysis
```bash
python main.py --seq-lengths 128 256 512 1024
```

---

## 📁 Project Structure

```
sparse-attention-viz/
├── main.py             # Training and evaluation runner
├── attention_viz.py    # Visualization functions
├── models.py           # Transformer models (Full, Random, Top-k)
├── utils.py            # Data loading, tokenizer, metrics
├── demo.ipynb          # Interactive Jupyter notebook
├── requirements.txt    # Dependencies
└── results/            # Generated on run
    ├── heatmaps/       # Attention heatmap PNGs
    ├── examples/       # Token selection images
    ├── interactive/    # Plotly HTML files
    └── metrics/        # JSON results
```

---

## 🎓 Learning Outcomes

After working through this project you will:
- ✅ Understand **visually** how sparse attention works
- ✅ See **experimentally** the full vs. sparse attention trade-offs
- ✅ Grasp the **core idea** behind MiniMax, Longformer, and BigBird
- ✅ Be able to **design** your own sparse attention variants

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas welcome:
- [ ] Threshold-based or learned-gate sparse selection
- [ ] Larger datasets (WikiText, BookCorpus)
- [ ] Multi-head attention visualization
- [ ] Real-time interactive dashboard

---

## 📚 References

- [MiniMax M3 Technical Report](https://www.minimax.ai/)
- [Sparse Transformer (OpenAI)](https://arxiv.org/abs/1904.10509)
- [Longformer](https://arxiv.org/abs/2004.05150)
- [BigBird](https://arxiv.org/abs/2007.14062)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Made with ❤️ and ☕ on CPU**

⭐ **Star this repo if you found it useful!** ⭐

</div>
