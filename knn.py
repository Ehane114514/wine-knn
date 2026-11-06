"""用 NumPy 实现的 K 近邻分类。"""

import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, features, labels):
        if not 1 <= self.k <= len(features):
            raise ValueError("k must be between 1 and the training set size")
        self.features = np.asarray(features, dtype=float)
        self.classes, self.label_ids = np.unique(labels, return_inverse=True)
        return self

    def predict(self, features):
        predictions = []
        for sample in features:
            # 平方欧氏距离与欧氏距离的排序相同，省去开平方。
            distances = np.sum((self.features - sample) ** 2, axis=1)
            neighbors = np.argsort(distances, kind="stable")[:self.k]
            votes = np.bincount(self.label_ids[neighbors], minlength=len(self.classes))
            # 若票数相同，选择排序靠前的类别，保证结果可复现。
            predictions.append(self.classes[votes.argmax()])
        return np.asarray(predictions)
