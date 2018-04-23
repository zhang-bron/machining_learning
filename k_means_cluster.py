# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import scale


def show_digits(digits_data, digits_labels):
    figure = plt.figure()
    for row in range(4):
        for col in range(4):
            ax = figure.add_subplot(4, 4, row * 4 + col + 1)
            img = digits_data[row * 4 + col].reshape(8, 8)
            label = digits_labels[row * 4 + col]

            ax.imshow(img, cmap=plt.cm.gray)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(label)
    plt.show()

# k_mean_model = KMeans(init='k-means++', n_clusters=10, n_init=10)
# k_mean_model.fit(data)
#
# label_pred = k_mean_model.predict(data)
#
# print(metrics.accuracy_score(label, label_pred))


digits = datasets.load_digits()
data = digits.data
labels = digits.target

show_digits(data, labels)

sample_count, feature_count = data.shape
digit_count = len(np.unique(labels))

print('数字：%d \t 样本量： %d \t 特征数：%d ' %(digit_count, sample_count, feature_count))

estimator = KMeans(init='k-means++', n_clusters=10, n_init=10)
estimator.fit(data)

print(estimator.labels_)

show_digits(digits.data, estimator.labels_)

print(estimator.inertia_)

print(metrics.homogeneity_score(labels, estimator.labels_))
print(metrics.completeness_score(labels, estimator.labels_))
print(metrics.v_measure_score(labels, estimator.labels_))
