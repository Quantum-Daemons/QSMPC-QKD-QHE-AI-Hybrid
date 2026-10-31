"""The full deployment's fraud endpoint scores the four columns the page's model reads.

WHY THIS EXISTS (Round 84). The bundled PaySim model now reads four of its eight sample columns (the
two balance-derived ones are a label leak). The API still fitted on all eight and padded or truncated
whatever it was given, so a four-value request was padded with zeros into the leaky columns and an
eight-value one kept them. It now fits on the four and selects them by name.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from api import inference_service  # noqa: E402
from qsh.models.registry import PAYSIM_MODEL_FEATURES, PAYSIM_SAMPLE_COLUMNS  # noqa: E402


def test_the_model_is_fitted_on_the_four_columns() -> None:
    inference_service._fraud_model.cache_clear()
    assert inference_service._fraud_model().coef_.shape[1] == len(PAYSIM_MODEL_FEATURES) == 4


def test_eight_columns_are_reduced_to_the_four_by_name() -> None:
    eight = [0.0] * len(PAYSIM_SAMPLE_COLUMNS)
    four = [0.0] * len(PAYSIM_MODEL_FEATURES)
    for i, name in enumerate(PAYSIM_MODEL_FEATURES):
        eight[PAYSIM_SAMPLE_COLUMNS.index(name)] = four[i] = 1.0 + i
    assert inference_service.predict_fraud(eight) == inference_service.predict_fraud(four)


def test_another_width_is_refused_not_padded() -> None:
    with pytest.raises(ValueError):
        inference_service.predict_fraud([1.0, 2.0, 3.0])
