"""运行 Wine 数据集的KNN 分类实验。"""

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from knn import KNN

matplotlib.use("Agg")
from matplotlib import pyplot as plt


SEED = 42
HERE = Path(__file__).resolve().parent


def scaled_split(train, validation):
    # 每折独立拟合标准化器，验证样本不参与计算均值和标准差。
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(validation)


def run_knn(features, labels):
    train, test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    ks = range(1, 16)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for fit_idx, val_idx in folds.split(train, y_train):
        fit_x, val_x = scaled_split(train[fit_idx], train[val_idx])
        fold_scores = []
        for k in ks:
            predicted = KNN(k).fit(fit_x, y_train[fit_idx]).predict(val_x)
            actual = y_train[val_idx]
            fold_scores.append((accuracy_score(actual, predicted),
                                f1_score(actual, predicted, average="macro")))
        scores.append(fold_scores)

    means = np.mean(scores, axis=0)
    # 先比较平均准确率，再比较宏平均 F1，仍相同则取较小的 k。
    best = max(ks, key=lambda k: (*np.round(means[k - 1], 12), -k))
    print(f"KNN 5-fold CV ({len(train)} train / {len(test)} test)")
    print(" k   accuracy  macro-F1")
    for k, (accuracy, f1) in zip(ks, means):
        print(f"{k:2d}   {accuracy:.6f}  {f1:.6f}")

    train_x, test_x = scaled_split(train, test)
    predicted = KNN(best).fit(train_x, y_train).predict(test_x)
    print(f"KNN selected k={best}")
    print(f"Test accuracy:        {accuracy_score(y_test, predicted):.6f}")
    print(f"Test macro precision: {precision_score(y_test, predicted, average='macro', zero_division=0):.6f}")
    print(f"Test macro recall:    {recall_score(y_test, predicted, average='macro'):.6f}")
    print(f"Test macro F1:        {f1_score(y_test, predicted, average='macro'):.6f}")

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(ks, means[:, 0], "o-", label="Mean CV accuracy")
    ax.axvline(best, color="tab:orange", linestyle="--", label=f"Selected k = {best}")
    ax.set(xlabel="Number of neighbors (k)", ylabel="Accuracy", title="KNN on Wine", xticks=list(ks))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(HERE / "knn_accuracy.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    wine = load_wine()
    run_knn(wine.data, wine.target)
