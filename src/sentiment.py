"""Evaluate TextBlob sentiment on TweetEval's supplied validation/test splits."""
import argparse
import html
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from textblob import TextBlob
from textblob.sentiments import PatternAnalyzer

from .common import ROOT, load_source, plot_style, save_run, write_json

LABELS = ['negative', 'neutral', 'positive']
THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
ANALYZER = PatternAnalyzer()


def clean_text(text):
    """Remove links/mentions; preserve negation, case, punctuation and hashtag words."""
    text = html.unescape(str(text))
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'(?<!\w)@\w+', ' ', text)
    text = re.sub(r'#(?=\w)', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def polarity(text):
    return float(TextBlob(clean_text(text), analyzer=ANALYZER).sentiment.polarity)


def classify(scores, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be between 0 and 1.')
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Polarity scores must be finite.')
    return np.where(scores > threshold, 2, np.where(scores < -threshold, 0, 1))


def choose_threshold(scores, labels):
    rows = [{'threshold': t, 'validation_macro_f1': float(f1_score(labels, classify(scores, t), labels=[0, 1, 2], average='macro', zero_division=0))} for t in THRESHOLDS]
    # Fixed ascending grid: ties choose the smaller threshold, without test labels.
    return max(rows, key=lambda row: row['validation_macro_f1'])['threshold'], pd.DataFrame(rows)


def load_split(split):
    text = load_source(f'{split}_text.txt').read_text(encoding='utf-8').splitlines()
    labels = np.array([int(x) for x in load_source(f'{split}_labels.txt').read_text(encoding='utf-8').splitlines()])
    if len(text) != len(labels) or not len(labels) or not set(labels).issubset({0, 1, 2}):
        raise ValueError(f'Invalid text/label alignment or labels in {split}.')
    return text, labels


def evaluate(labels, predicted):
    report = classification_report(labels, predicted, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(labels, predicted)), 'macro_precision': float(report['macro avg']['precision']),
            'macro_recall': float(report['macro avg']['recall']), 'macro_f1': float(report['macro avg']['f1-score']),
            'classification_report': report}


def run():
    # Only train labels are needed for the majority baseline; TextBlob is not fitted.
    train_labels = np.array([int(x) for x in load_source('train_labels.txt').read_text(encoding='utf-8').splitlines()])
    if not len(train_labels) or not set(train_labels).issubset({0, 1, 2}):
        raise ValueError('Invalid training labels.')
    val_text, val_labels = load_split('val')
    val_scores = np.array([polarity(text) for text in val_text])
    threshold, search = choose_threshold(val_scores, val_labels)
    # Read/evaluate test labels only after selecting the threshold on validation.
    test_text, test_labels = load_split('test')
    test_scores = np.array([polarity(text) for text in test_text])
    predictions = classify(test_scores, threshold)
    majority = int(np.bincount(train_labels, minlength=3).argmax())
    metrics = {'training_labels_for_baseline': len(train_labels), 'validation_tweets': len(val_labels),
               'test_tweets': len(test_labels), 'selected_threshold': threshold,
               'selection_metric': 'validation macro F1; fixed grid; smallest threshold breaks ties',
               'validation_macro_f1': float(search['validation_macro_f1'].max()),
               'test': evaluate(test_labels, predictions),
               'training_majority_label': LABELS[majority],
               'majority_baseline_test': evaluate(test_labels, np.full(len(test_labels), majority)),
               'empty_after_cleaning_test': sum(not clean_text(t) for t in test_text),
               'test_cleaned_text_duplicates': int(pd.Series([clean_text(t) for t in test_text]).duplicated().sum()),
               'note': 'Fixed lexicon baseline, not a newly trained NLP model. Polarity is not probability. Official supplied splits retained.'}
    reports = save_run(metrics, ['pandas', 'numpy', 'matplotlib', 'scikit-learn', 'textblob'])
    search.to_csv(reports / 'threshold_search.csv', index=False)
    pd.DataFrame({'test_row': range(len(test_labels)), 'gold_label': [LABELS[i] for i in test_labels],
                  'prediction': [LABELS[i] for i in predictions], 'polarity': test_scores}).to_csv(reports / 'test_predictions.csv', index=False)
    write_json(reports / 'config.json', {'threshold': threshold, 'analyzer': 'TextBlob PatternAnalyzer', 'label_order': LABELS})
    matrix = confusion_matrix(test_labels, predictions, labels=[0, 1, 2])
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(reports / 'confusion_matrix.csv', index_label='actual')
    plot_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.5), layout='constrained')
    ax.grid(False)
    im = ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=range(3), xticklabels=LABELS, yticks=range(3), yticklabels=LABELS,
           xlabel='Predicted sentiment', ylabel='Actual sentiment', title=f'TextBlob on {len(test_labels):,} held-out tweets')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i, j]:,}', ha='center', va='center', fontsize=13,
                    color='white' if matrix[i, j] > matrix.max() / 2 else '#23344d')
    fig.colorbar(im, ax=ax, label='Tweet count')
    fig.savefig(reports / 'confusion_matrix.png', bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.3), layout='constrained')
    positions = np.arange(3)
    ax.bar(positions - .18, np.bincount(test_labels, minlength=3), width=.36, color='#007f86', label='Actual')
    ax.bar(positions + .18, np.bincount(predictions, minlength=3), width=.36, color='#e48b32', label='Predicted')
    ax.set(xticks=positions, xticklabels=LABELS, ylabel='Tweets', title='Class balance and prediction bias')
    ax.legend(frameon=False)
    fig.savefig(reports / 'sentiment_distribution.png')
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    return metrics


def analyze(text):
    config = ROOT / 'reports/config.json'
    if not config.exists():
        raise FileNotFoundError('Run python -m src.sentiment first.')
    threshold = json.loads(config.read_text(encoding='utf-8'))['threshold']
    score = polarity(text)
    return {'cleaned_text': clean_text(text), 'polarity': score, 'sentiment': LABELS[int(classify([score], threshold)[0])],
            'threshold': threshold, 'note': 'Sentiment polarity, not emotion or confidence.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', help='Analyze one text using the saved validation-selected threshold')
    args = parser.parse_args()
    if args.text is None:
        run()
    else:
        print(json.dumps(analyze(args.text), indent=2))
