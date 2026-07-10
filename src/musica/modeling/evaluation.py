"""Model evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, f1_score

from musica.logging import logger


@dataclass(frozen=True)
class EvaluationResult:
    test_loss: float
    test_accuracy: float
    f1_macro: float
    report: str


class ChordEvaluator:
    def evaluate(
            self,
            model: Any,
            x_test: np.ndarray,
            y_test: np.ndarray,
            labels: list[str],
    ) -> EvaluationResult:
        logger.info("Evaluation sur le test: {} exemples", x_test.shape[0])
        y_pred_proba = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        result = EvaluationResult(
            test_loss=float(test_loss),
            test_accuracy=float(test_accuracy),
            f1_macro=float(f1_score(y_test, y_pred, average="macro")),
            report=classification_report(y_test, y_pred, target_names=labels),
        )
        logger.info(
            "Evaluation terminee: accuracy={:.4f}, loss={:.4f}, f1_macro={:.4f}",
            result.test_accuracy,
            result.test_loss,
            result.f1_macro,
        )
        return result
