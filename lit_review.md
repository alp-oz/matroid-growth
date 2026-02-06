# Matroid Model Reproduction

## 1. The Greedy Baseline
- **Source:** Jack Edmonds (1971)
- **Logic:** Add elements to the basis until the rank is full.
- **Key Feature:** Static. Once full, it ignores all new inputs.

## 2. The Basis Exchange Model
- **Source:** Basis Exchange Property (Classical Matroid Theory)
- **Logic:** After the rank is full, new elements can "swap" with old ones.
- **Key Feature:** Dynamic. The basis evolves over time via XOR pivots.