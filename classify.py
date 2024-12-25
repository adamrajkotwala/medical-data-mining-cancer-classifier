import pandas as pd
import numpy as np
from collections import Counter
from joblib import Parallel, delayed
from numba import jit
import random
import time
import math
import matplotlib.pyplot as plt

start_time = time.time()

df = pd.read_csv('mutations.csv', index_col=0)
data = df.to_numpy()
index = df.index.to_numpy()
columns = df.columns.to_numpy()
is_cancer_global = np.array([idx.startswith('C') for idx in index])
num_trees = 1000
is_cancer_global = np.array([idx.startswith('C') for idx in index])

# data: NumPy array containing the dataset.
# index: NumPy array containing the sample labels.
# returns: a tuple with two elements:
#          - The first element is another tuple containing the bootstrapped data (NumPy array) and indices.
#          - The second element is a tuple containing the out-of-bag data (NumPy array) and indices.
def bootstrap(data, index):
    n_samples = len(index)
    indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
    bootstrapped_data = data[indices]
    bootstrapped_index = index[indices]
    out_of_bag_mask = np.ones(n_samples, dtype=bool)
    out_of_bag_mask[indices] = False
    out_of_bag_data = data[out_of_bag_mask]
    out_of_bag_index = index[out_of_bag_mask]
    return (bootstrapped_data, bootstrapped_index), (out_of_bag_data, out_of_bag_index)

# is_cancer: NumPy array indicating whether each sample is cancerous (True/False).
# feature_values: NumPy array containing the values of a specific feature across samples.
# returns: a float value representing the quality of the split.
@jit(nopython=True)
def compute_phi_numba(is_cancer, feature_values):
    n_t = len(is_cancer)
    left_mask = feature_values == 1
    right_mask = ~left_mask

    n_tL = np.sum(left_mask)
    n_tR = n_t - n_tL
    n_tL_C = np.sum(is_cancer & left_mask)
    n_tR_C = np.sum(is_cancer & right_mask)

    P_L = n_tL / n_t if n_t > 0 else 0
    P_R = n_tR / n_t if n_t > 0 else 0
    Q_s_t = abs(n_tL_C / n_tL - n_tR_C / n_tR) if n_tL > 0 and n_tR > 0 else 0
    phi_s_t = 2 * P_L * P_R * Q_s_t

    return phi_s_t

# data: NumPy array containing the dataset.
# is_cancer: NumPy array indicating whether each sample is cancerous (True/False).
# feature_idx: integer index of the feature to evaluate.
# returns: a float value representing the quality of the split.
def compute_phi(data, is_cancer, feature_idx):
    feature_values = data[:, feature_idx]
    return compute_phi_numba(is_cancer, feature_values)

# data: NumPy array containing the dataset.
# index: NumPy array containing the sample labels.
# max_depth: maximum depth of the decision tree (default: 17).
# returns: a dictionary representing the decision tree structure.
def build_decision_tree(data, index, max_depth=17):
    def create_subtree(data, is_cancer, depth):
        if depth >= max_depth or data.shape[0] == 0:
            num_cancer = np.sum(is_cancer)
            num_non_cancer = len(is_cancer) - num_cancer
            return {'label': 'C' if num_cancer >= num_non_cancer else 'NC'}

        num_features = data.shape[1]
        num_features_to_sample = math.ceil(num_features / 4)
        feature_indices = random.sample(range(num_features), num_features_to_sample)

        phi_values = {feature_idx: compute_phi(data, is_cancer, feature_idx) for feature_idx in feature_indices}
        best_feature_idx = max(phi_values, key=phi_values.get)

        left_mask = data[:, best_feature_idx] == 1
        right_mask = ~left_mask

        left_data = data[left_mask]
        right_data = data[right_mask]
        left_is_cancer = is_cancer[left_mask]
        right_is_cancer = is_cancer[right_mask]

        return {
            'feature': best_feature_idx,
            'left': create_subtree(left_data, left_is_cancer, depth + 1),
            'right': create_subtree(right_data, right_is_cancer, depth + 1)
        }

    is_cancer = np.array([idx.startswith('C') for idx in index])
    return create_subtree(data, is_cancer, 0)

# data: NumPy array containing the dataset.
# index: NumPy array containing the sample labels.
# num_trees: number of decision trees to build.
# max_depth: maximum depth of each decision tree (default: 20).
# returns: a list of dictionaries, each representing a decision tree.
def build_forest(data, index, num_trees, max_depth=17):
    bootstrapped_samples = Parallel(n_jobs=-1)(
        delayed(bootstrap)(data, index) for _ in range(num_trees)
    )
    forest = Parallel(n_jobs=-1)(
        delayed(build_decision_tree)(sample[0][0], sample[0][1], max_depth=max_depth) for sample in bootstrapped_samples
    )
    return forest

# tree: dictionary representing a decision tree.
# sample: NumPy array representing a single sample with features.
# returns: the label ('C' or 'NC') predicted by the tree for the sample.
def classify_sample(tree, sample):
    while 'label' not in tree:
        if sample[tree['feature']] == 1:
            tree = tree['left']
        else:
            tree = tree['right']
    return tree['label']

# forest: list of decision tree dictionaries.
# sample: NumPy array representing a single sample with features.
# returns: a tuple containing:
#          - Predicted label ('C' or 'NC').
#          - Count of votes for 'C'.
#          - Count of votes for 'NC'.
def random_forest_classifier(forest, sample):
    cancer_votes = 0
    non_cancer_votes = 0

    for tree in forest:
        prediction = classify_sample(tree, sample)
        if prediction == 'C':
            cancer_votes += 1
        else:
            non_cancer_votes += 1

    return 'C' if cancer_votes > non_cancer_votes else 'NC', cancer_votes, non_cancer_votes

# data: NumPy array containing the dataset.
# index: NumPy array containing the sample labels.
# decision_tree: dictionary representing a single decision tree.
# returns: a dictionary with counts for:
#          - True Positives (TP)
#          - False Positives (FP)
#          - True Negatives (TN)
#          - False Negatives (FN).
def mutation_confusion_matrix(data, index, decision_tree):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for i in range(len(index)):
        is_cancer = index[i].startswith('C')
        prediction = classify_sample(decision_tree, data[i])
        if is_cancer and prediction == 'C':
            true_positive += 1
        elif is_cancer and prediction == 'NC':
            false_negative += 1
        elif not is_cancer and prediction == 'C':
            false_positive += 1
        else:
            true_negative += 1
    return {
        "True Positives": true_positive,
        "False Negatives": false_negative,
        "False Positives": false_positive,
        "True Negatives": true_negative
    }

# confusion_matrix: dictionary containing counts for TP, FP, TN, and FN.
# returns: a dictionary with calculated metrics:
#          - Accuracy
#          - Sensitivity
#          - Specificity
#          - Precision.
def calculate_metrics(confusion_matrix):
    TP = confusion_matrix["True Positives"]
    TN = confusion_matrix["True Negatives"]
    FP = confusion_matrix["False Positives"]
    FN = confusion_matrix["False Negatives"]

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
    }

# data: NumPy array containing the dataset.
# index: NumPy array containing the sample labels.
# min_cancer_coverage: minimum number of cancer samples required for a feature (default: 3).
# max_non_cancer_coverage: maximum number of non-cancer samples allowed for a feature (default: 70).
# returns: a tuple containing:
#          - Filtered data (NumPy array).
#          - List of indices for the selected features.
def filter_features(data, index, min_cancer_coverage, max_non_cancer_coverage):
    is_cancer = np.array([idx.startswith('C') for idx in index])
    feature_cancer_coverage = []
    feature_non_cancer_coverage = []

    for feature_idx in range(data.shape[1]):
        feature_values = data[:, feature_idx] # selects all rows at column feature_idx
        cancer_coverage = np.sum(feature_values[is_cancer])
        non_cancer_coverage = np.sum(feature_values[~is_cancer])
        feature_cancer_coverage.append(cancer_coverage)
        feature_non_cancer_coverage.append(non_cancer_coverage)

    features_to_keep = [                                 # zip combines two lists 'feature_cancer_coverage', 'feature_non_cancer_coverage' into pairs of tuples
        idx for idx, (cancer_cov, non_cancer_cov) in enumerate(zip(feature_cancer_coverage, feature_non_cancer_coverage)) # enumerate adds an index to each tuple in the `zip` result
        if cancer_cov >= min_cancer_coverage and non_cancer_cov <= max_non_cancer_coverage
    ]

    filtered_data = data[:, features_to_keep] # stores all of the rows (sample information) for each of the retained features
    return filtered_data, features_to_keep

# index: NumPy array containing the sample labels.
# is_cancer: NumPy array indicating whether each sample is cancerous (True/False).
# predictions: list of predicted labels ('C' or 'NC') for all samples.
# returns: None. Displays a bar plot showing misclassification analysis.
def plot_misclassification(index, is_cancer, predictions):
    misclassified = [idx for i, idx in enumerate(index) if predictions[i] != ('C' if is_cancer[i] else 'NC')]
    cancer_misclassified = [idx for idx in misclassified if idx.startswith('C')]
    non_cancer_misclassified = [idx for idx in misclassified if not idx.startswith('C')]

    labels = ['Correctly Classified', 'Misclassified']
    counts = [len(index) - len(misclassified), len(misclassified)]

    print(f"\nMisclassified Cancer Samples: {cancer_misclassified}")
    print(f"Misclassified Non-Cancer Samples: {non_cancer_misclassified}\n")
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=['green', 'red'])
    plt.title('Overall Classification Results')
    plt.ylabel('Number of Samples')
    plt.show()

# forest: list of decision tree dictionaries.
# filtered_columns: list of column names corresponding to the features in the dataset.
# returns: None. Displays a horizontal bar plot of the top 10 most important features.
def plot_feature_importance(forest, filtered_columns):
    feature_importance = Counter()
    for tree in forest:
        def traverse_tree(tree):
            if 'feature' in tree:
                feature_importance[tree['feature']] += 1
                traverse_tree(tree['left'])
                traverse_tree(tree['right'])
        traverse_tree(tree)

    important_features = {filtered_columns[idx]: count for idx, count in feature_importance.items()}
    sorted_features = sorted(important_features.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:10]
    truncated_feature_names = [name[:15] for name, _ in top_features]
    importance_counts = [count for _, count in top_features]

    print("\nTop Features and Importance Counts:\n")
    for feature, count in top_features:
        print(f"{feature}: {count}")
    
    plt.figure(figsize=(12, 6))
    plt.barh(truncated_feature_names, importance_counts, color='blue')
    plt.xlabel('Feature Importance (Number of Splits)')
    plt.title('Top 10 Important Features')
    plt.gca().invert_yaxis()
    plt.show()

# Apply feature filtering
filtered_data, selected_features = filter_features(data, index, min_cancer_coverage=1, max_non_cancer_coverage = 10)
print(f"Selected {len(selected_features)} features out of {data.shape[1]} total features.")

# Get filtered column names
filtered_columns = columns[selected_features]

# Build forest
forest = build_forest(filtered_data, index, num_trees)

# Classify all samples
predictions = []
for i in range(len(index)):
    sample = filtered_data[i]
    prediction, _, _ = random_forest_classifier(forest, sample)
    predictions.append(prediction)

# Classify specific samples
specific_samples = ['C1', 'C10', 'C15', 'NC5', 'NC15']
for sample_id in specific_samples:
    sample_idx = np.where(index == sample_id)[0][0]
    sample = filtered_data[sample_idx]
    classification, cancer_votes, non_cancer_votes = random_forest_classifier(forest, sample)
    print(f"Sample {sample_id}: Classification: {classification}, Cancer Votes: {cancer_votes}, Non-Cancer Votes: {non_cancer_votes}")

# Confusion matrix and metrics
first_tree = forest[0]
confusion_matrix = mutation_confusion_matrix(filtered_data, index, first_tree)
metrics = calculate_metrics(confusion_matrix)

print("\nConfusion Matrix for the First Tree:\n")
for key, value in confusion_matrix.items():
    print(f"{key}: {value}")

print("\nMetrics for the First Tree:\n")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

plot_misclassification(index, is_cancer_global, predictions)
plot_feature_importance(forest, filtered_columns)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time:.2f} seconds")
