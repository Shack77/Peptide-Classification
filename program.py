# Improved NumPy FFNN with k-mer features, deeper network, better initialization, and weighted loss
import numpy as np
from collections import Counter
from sklearn.metrics import matthews_corrcoef
import os
import itertools

# -------------------------------
# Load and preprocess the data
# -------------------------------
def load_data(train_file, test_file):
    with open(train_file, 'r') as f:
        train_lines = f.readlines()
    with open(test_file, 'r') as f:
        test_lines = f.readlines()

    train_labels, train_sequences = [], []
    for line in train_lines:
        label, sequence = line.strip().split('\t')
        train_labels.append(int(label))
        train_sequences.append(sequence)

    test_sequences = [line.strip() for line in test_lines]
    return train_labels, train_sequences, test_sequences

# -------------------------------
# Feature Extraction: K-mers
# -------------------------------
def extract_kmer_features(sequences, k=2):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    kmers = [''.join(p) for p in itertools.product(amino_acids, repeat=k)]
    kmer_index = {kmer: idx for idx, kmer in enumerate(kmers)}

    features = []
    for seq in sequences:
        vec = np.zeros(len(kmers))
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if kmer in kmer_index:
                vec[kmer_index[kmer]] += 1
        features.append(vec)
    return np.array(features)

# -------------------------------
# Activation Functions
# -------------------------------
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1 - np.tanh(x) ** 2

# -------------------------------
# Improved Feedforward Network
# -------------------------------
def train_nn_improved(X_train, y_train, hidden1=128, hidden2=64, hidden3=32, epochs=500, lr=0.05, w0=1.0, w1=1.0):
    input_dim, output_dim = X_train.shape[1], 1
    m = X_train.shape[0]

    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2. / input_dim)
    b1 = np.zeros((1, hidden1))
    W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2. / hidden1)
    b2 = np.zeros((1, hidden2))
    W3 = np.random.randn(hidden2, hidden3) * np.sqrt(2. / hidden2)
    b3 = np.zeros((1, hidden3))
    W4 = np.random.randn(hidden3, output_dim) * np.sqrt(2. / hidden3)
    b4 = np.zeros((1, output_dim))

    y_bin = ((y_train + 1) // 2).reshape(-1, 1)
    losses = []

    for epoch in range(epochs):
        Z1 = np.dot(X_train, W1) + b1
        A1 = tanh(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = tanh(Z2)
        Z3 = np.dot(A2, W3) + b3
        A3 = tanh(Z3)
        Z4 = np.dot(A3, W4) + b4
        A4 = sigmoid(Z4)

        weights = np.where(y_bin == 1, w1, w0)
        loss = -np.mean(weights * (y_bin * np.log(A4 + 1e-8) + (1 - y_bin) * np.log(1 - A4 + 1e-8)))
        losses.append(loss)

        dZ4 = A4 - y_bin
        dW4 = np.dot(A3.T, dZ4) / m
        db4 = np.sum(dZ4, axis=0, keepdims=True) / m

        dA3 = np.dot(dZ4, W4.T)
        dZ3 = dA3 * tanh_derivative(Z3)
        dW3 = np.dot(A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, W3.T)
        dZ2 = dA2 * tanh_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * tanh_derivative(Z1)
        dW1 = np.dot(X_train.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3
        W4 -= lr * dW4
        b4 -= lr * db4

    return W1, b1, W2, b2, W3, b3, W4, b4, losses

# -------------------------------
# Prediction
# -------------------------------
def predict(X, W1, b1, W2, b2, W3, b3, W4, b4):
    A1 = tanh(np.dot(X, W1) + b1)
    A2 = tanh(np.dot(A1, W2) + b2)
    A3 = tanh(np.dot(A2, W3) + b3)
    A4 = sigmoid(np.dot(A3, W4) + b4)
    return (A4 >= 0.5).astype(int)

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    train_file = "train.dat"
    test_file = "test.dat"

    train_labels, train_sequences, test_sequences = load_data(train_file, test_file)
    X_train = extract_kmer_features(train_sequences, k=2)
    X_test = extract_kmer_features(test_sequences, k=2)
    y_train = np.array(train_labels)

    N, N_pos, N_neg = len(y_train), sum(y_train == 1), sum(y_train == -1)
    w1, w0 = N / (2 * N_pos), N / (2 * N_neg)

    W1, b1, W2, b2, W3, b3, W4, b4, _ = train_nn_improved(X_train, y_train, w0=w0, w1=w1)

    test_preds = predict(X_test, W1, b1, W2, b2, W3, b3, W4, b4)
    output_path = os.path.abspath("submission.txt")
    with open(output_path, "w") as f:
        for pred in test_preds:
            f.write(f"{1 if pred[0] == 1 else -1}\n")
    print("Predictions saved to:", output_path)

    train_preds = predict(X_train, W1, b1, W2, b2, W3, b3, W4, b4)
    train_preds_signed = np.where(train_preds == 1, 1, -1)
    print("Training MCC:", matthews_corrcoef(y_train, train_preds_signed))