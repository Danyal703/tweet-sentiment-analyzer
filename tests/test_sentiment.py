import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score

from src.common import ROOT
from src.sentiment import classify, clean_text, choose_threshold, polarity


def test_cleaner_preserves_negation_and_emphasis():
    assert clean_text('@user This is NOT good! #delayed https://example.org') == 'This is NOT good! delayed'
    assert clean_text('A &amp; B') == 'A & B'


def test_threshold_boundaries_are_neutral():
    assert classify([-.21, -.2, 0, .2, .21], .2).tolist() == [0, 1, 1, 1, 2]


def test_validation_selection_can_choose_nonzero_threshold():
    threshold, table = choose_threshold([-.8, -.1, .1, .8], [0, 1, 1, 2])
    assert threshold == .1
    assert table['validation_macro_f1'].max() == 1


def test_empty_text_is_neutral_not_missing_class():
    assert polarity('') == 0
    assert classify([polarity('@user https://example.org')], .2).tolist() == [1]


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        classify([0], float('nan'))


def test_saved_metrics_recompute_from_all_test_predictions():
    predictions = pd.read_csv(ROOT / 'reports/test_predictions.csv')
    metrics = json.loads((ROOT / 'reports/metrics.json').read_text())
    assert len(predictions) == metrics['test_tweets']
    assert predictions['test_row'].tolist() == list(range(len(predictions)))
    assert f1_score(predictions.gold_label, predictions.prediction, average='macro') == pytest.approx(metrics['test']['macro_f1'])
    assert precision_score(predictions.gold_label, predictions.prediction, average='macro') == pytest.approx(metrics['test']['macro_precision'])
