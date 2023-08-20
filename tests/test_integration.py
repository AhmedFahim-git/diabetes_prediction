from __future__ import annotations

import pytest
from sklearn.metrics import accuracy_score

from predictions import predict


@pytest.fixture
def all_data():
    return predict.get_data()


def test_get_predictions(all_data):
    results = predict.make_predictions(all_data)
    accuracy = accuracy_score(y_true=results["Outcome"], y_pred=results["predictions"])
    assert (
        accuracy > 0.8
    ), "We should be able to get the proper predictions and the \
        accuracy should be above 80 percent"
