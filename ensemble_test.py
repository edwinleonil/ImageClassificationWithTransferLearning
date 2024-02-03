import pandas as pd
import numpy as np
import os

BASE_PATH = r"C:\test_results"
NUM_CLASSES = 9

FILE_NAMES = [
    f"InceptionV3-200525-{NUM_CLASSES}classes-run-1.csv",
    f"GoogleNet-200525-{NUM_CLASSES}classes-run-1.csv",
    f"ResNet18-200525-{NUM_CLASSES}classes-run-1.csv",
    f"ResNet50-200525-{NUM_CLASSES}classes-run-1.csv",
    f"ResNet101-200525-{NUM_CLASSES}classes-run-1.csv",
    f"ViT_base_patch16_224-200525-{NUM_CLASSES}classes-run-1.csv",
    f"Xception-200525-{NUM_CLASSES}classes-run-1.csv"
]

PROB_PATHS = [os.path.join(BASE_PATH, file_name) for file_name in FILE_NAMES]

def load_dataframes(prob_paths):
    correct_predictions = []
    incorrect_predictions = []

    for path in prob_paths:
        df = pd.read_csv(path)
        correct_predictions.append(df[df.iloc[:, 1] == df.iloc[:, 2]])
        incorrect_predictions.append(df[df.iloc[:, 1] != df.iloc[:, 2]])

    return correct_predictions, incorrect_predictions

def calculate_avg_prob(df):
    true_labels = df.iloc[:, 1]
    probabilities = df.iloc[:, 3:]
    max_prob = probabilities.max(axis=1)
    avg_max_prob = np.array([max_prob[true_labels == label].mean() for label in true_labels.unique()])
    return np.c_[true_labels.unique(), avg_max_prob]

def add_missing_classes(correct_probs, incorrect_probs):
    for i in range(len(correct_probs)):
        missing_classes = np.setdiff1d(correct_probs[i][:, 0], incorrect_probs[i][:, 0])
        for missing_class in missing_classes:
            incorrect_probs[i] = np.vstack((incorrect_probs[i], np.array([missing_class, 0.0])))
    return correct_probs, incorrect_probs

def convert_and_sort_probs(probs):
    for i in range(len(probs)):
        probs[i][:, 1] = probs[i][:, 1].astype(float)
        probs[i] = probs[i][probs[i][:, 0].argsort()]
    return probs

def calculate_diffs(correct_probs, incorrect_probs):
    return [correct_probs[i][:, 1] - incorrect_probs[i][:, 1] for i in range(len(correct_probs))]

def calculate_weighted_probs(prob_paths, diffs):
    weighted_probs = []
    for i in range(len(prob_paths)):
        df = pd.read_csv(prob_paths[i])
        probabilities = df.iloc[:, 3:]
        weighted_probs.append(probabilities * diffs[i])
    return weighted_probs

def calculate_corrected_prob(weighted_probs):
    corrected_prob = weighted_probs[0]
    for i in range(1, len(weighted_probs)):
        corrected_prob = corrected_prob.add(weighted_probs[i])
    return corrected_prob / len(weighted_probs)

def normalize_and_round(corrected_prob):
    corrected_prob = corrected_prob.div(corrected_prob.sum(axis=1), axis=0)
    return corrected_prob.astype(float).round(2)

def add_columns(corrected_prob, df, col_names):
    predicted_class = corrected_prob.idxmax(axis=1)
    corrected_prob.insert(0, col_names[0], df.iloc[:, 0])
    corrected_prob.insert(1, col_names[1], df.iloc[:, 1])
    corrected_prob.insert(2, col_names[2], predicted_class)
    return corrected_prob

def save_to_csv(df, path):
    df.to_csv(path, index=False)

def main():
    df = pd.read_csv(PROB_PATHS[0])
    correct_predictions, incorrect_predictions = load_dataframes(PROB_PATHS)
    correct_probs = [calculate_avg_prob(df) for df in correct_predictions]
    incorrect_probs = [calculate_avg_prob(df) for df in incorrect_predictions]
    correct_probs, incorrect_probs = add_missing_classes(correct_probs, incorrect_probs)
    correct_probs = convert_and_sort_probs(correct_probs)
    incorrect_probs = convert_and_sort_probs(incorrect_probs)
    diffs = calculate_diffs(correct_probs, incorrect_probs)
    weighted_probs = calculate_weighted_probs(PROB_PATHS, diffs)
    corrected_prob = calculate_corrected_prob(weighted_probs)
    corrected_prob = normalize_and_round(corrected_prob)
    corrected_prob = add_columns(corrected_prob, df, df.columns[:3])
    save_to_csv(corrected_prob, os.path.join(BASE_PATH, "EnsembleModel_test.csv"))

if __name__ == "__main__":
    main()