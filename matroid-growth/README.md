# Matroid Growth & Phase Transitions

A research-oriented simulator for observing the emergence of algebraic minors in growing scale-free binary matroids.

## 🔬 Research Objective
This project explores the boundary between **Randomness** and **Algebraic Order**. It simulates a binary matrix that grows over time, where new columns are added via a "Preferential Attachment" mechanism (Barabási-Albert model). 

The goal is to identify the critical thresholds ($\beta, \gamma$) where the matrix stops being a "Free Matroid" and begins to contain specific forbidden minors, such as the **Fano Plane ($F_7$)**.

## 🏗 Project Architecture
The project is modularized to ensure scientific reproducibility:
- `matroid_core/`: Contains the stochastic engine, binary rank logic, and minor detection algorithms.
- `main.py`: Interactive dashboard for single-run experiments.
- `batch_runner.py`: Tool for 1D parameter sweeps (Phase Transition S-curves).
- `heatmap_runner.py`: 2D parameter sweep for mapping the Phase Frontier.
- `experiments/`: Automated logging of every simulation to CSV for data analysis.

## 📊 Key Metrics
- **Matroid Rank $r(M)$:** Calculated over $GF(2)$ via Gaussian Elimination.
- **Independence Ratio:** The ratio of $r(M)$ to Row Count. A drop in this ratio signals "Structural Compression."
- **Minor Detection:** Search algorithms for $F_7$ (Fano) and $W_3$ (Wheel) minors.

## 🚀 Getting Started

1. **Install Dependencies:**
   ```bash
   pip install numpy matplotlib seaborn scipy
2. Run a Single Simulation:

Bash
python main.py
3. Generate Research Data (Heatmap):

Bash
python heatmap_runner.py

### 3. Final Deployment Checklist
Before you push to GitHub, make sure you've handled these "cleaning" steps:

1.  **`.gitignore`**: Create a file named `.gitignore` and add these lines so you don't upload temporary junk:
    ```text
    __pycache__/
    *.pyc
    experiments/*.png
    experiments/research_log.csv
    .DS_Store
    ```
2.  **`requirements.txt`**: Create this file so others can install your setup:
    ```text
    numpy
    matplotlib
    seaborn
    scipy
    ```

### Next Step for you:
Since you now have a research journal (`research_log.csv`), would you like me to help you create a **Data Analysis Notebook** (or script) that automatically calculates the **Correlation Coefficient** between the "Independence Ratio" and the "Fano Probability"? 

This "Correlation" is the single strongest piece of evidence you can put in a grant to prove your theory is correct.