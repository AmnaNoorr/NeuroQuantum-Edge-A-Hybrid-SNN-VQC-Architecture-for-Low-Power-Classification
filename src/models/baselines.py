"""Classical sklearn baselines used in the ECG comparison protocol."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


class LogisticRegressionBaseline(ClassifierMixin, BaseEstimator):
    """A reproducible sklearn logistic-regression binary classification baseline.

    The estimator follows sklearn's ``fit``/``predict`` interface and exposes
    :attr:`parameter_count` after fitting. The count includes every learned
    coefficient and intercept term, allowing direct reporting beside neural
    model parameter counts.

    Args:
        c: Inverse regularization strength passed to sklearn.
        max_iter: Maximum optimization iterations.
        random_state: Seed used by solvers with randomized behavior.
        class_weight: Optional sklearn class-weight policy.
        solver: sklearn logistic-regression solver.
    """

    def __init__(
        self,
        *,
        c: float = 1.0,
        max_iter: int = 1_000,
        random_state: int | None = 42,
        class_weight: str | dict[int, float] | None = None,
        solver: str = "lbfgs",
    ) -> None:
        """Store sklearn-compatible hyperparameters without fitting data."""
        self.c = c
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.solver = solver

    def fit(self, features: ArrayLike, labels: ArrayLike) -> "LogisticRegressionBaseline":
        """Fit the baseline on standardized training features only.

        Args:
            features: Two-dimensional training feature matrix.
            labels: Binary training labels.

        Returns:
            This fitted estimator.
        """
        self.estimator_ = LogisticRegression(
            C=self.c,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
            solver=self.solver,
        )
        self.estimator_.fit(features, labels)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        """Predict normal/abnormal class labels for feature rows."""
        return self._fitted_estimator().predict(features)

    def predict_proba(self, features: ArrayLike) -> np.ndarray:
        """Return normal/abnormal class probabilities for feature rows."""
        return self._fitted_estimator().predict_proba(features)

    @property
    def parameter_count(self) -> int:
        """Return the fitted coefficient and intercept count.

        Raises:
            sklearn.exceptions.NotFittedError: If called before :meth:`fit`.
        """
        estimator = self._fitted_estimator()
        return int(estimator.coef_.size + estimator.intercept_.size)

    def _fitted_estimator(self) -> LogisticRegression:
        """Return the trained sklearn estimator or raise its standard error."""
        check_is_fitted(self, "estimator_")
        return self.estimator_
