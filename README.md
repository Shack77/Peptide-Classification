# Peptide-Classification
###README — Antibiofilm Peptide Classifier using NumPy

####Project Overview

This project implements a **feed-forward neural network from scratch using NumPy** to predict whether a given **antibacterial peptide** is also **antibiofilm**. The task is a binary classification problem.

---

####Dataset Format

* **train.dat**: Each line is formatted as
  `label<TAB>peptide_sequence`
  where `label ∈ {1, -1}`

* **test.dat**: Contains only peptide sequences, one per line.

---

#### Key Features

* **Custom Neural Network**: 3 hidden layers (128 → 64 → 32), trained via backpropagation.
* **K-mer Encoding (k=2)**: Captures short patterns of amino acid residues.
* **Class Imbalance Handling**: Uses class-weighted binary cross-entropy loss.
* **Matthews Correlation Coefficient (MCC)** for evaluation.

---

#### Requirements

* Python 3.x
* NumPy
* scikit-learn (for MCC)

Install dependencies:

```bash
pip install numpy scikit-learn
```

---

#### How to Run

1. Place `train.dat` and `test.dat` in the same directory.
2. Run the script:

```bash
python your_script_name.py
```

3. Outputs:

   * `submission.txt`: Predictions for test data (`+1` or `-1`)
   * Console logs: Final MCC score on training data

---

#### Output Format

* Each line in `submission.txt` is either `+1` or `-1`, one per test sample.

---

#### Tips to Improve Performance

* Try `k=3` in `extract_kmer_features()` for richer features.
* Add L2 regularization or dropout.
* Implement learning rate decay.
* Consider adding a validation set and tracking MCC during training.
