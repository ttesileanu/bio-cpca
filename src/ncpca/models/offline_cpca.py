import numpy as np
import scipy.linalg as scla

from typing import Optional


class OfflineCPCA:
    """Perform contrastive PCA using offline algorithms."""

    def __init__(
        self,
        n_components: int = 2,
        beta: float = 0.5,
        center: bool = True,
        normalize: bool = True,
        variant: str = "star",
    ):
        """Contrastive Principal Component Analysis (cPCA) using a variation on the
        original algorithm from Abid et al. See https://arxiv.org/abs/2211.07723.

        For one component, the algorithm can be defined in terms of finding the
        projection that maximizes the ratio between the positive- and negative-sample
        variances. Shrinkage can be applied on the negative samples; see `beta`.

        The assumption is that the positive samples contain some interesting directions
        of variation (e.g., separate different classes of interest) but are contaminated
        by variability in uninteresting directions. In contrast, the negative samples
        contain only the uninteresting variations. cPCA uses the information from the
        negative samples to discount the uninteresting variations and focus on the
        interesting ones.

        In general, the contrastive principal components are found by solving a
        generalized eigenvalue problem:
            C_fg @ v = w * C_bg_shrunk @ v
        where `w` is the generalized eigenvalue, `v` the generalized eigenvector, `C_fg`
        the covariance of the positive samples, and `C_bg_shrunk` is given by
            C_bg_shrunk = (1 - beta) * eye + beta * C_bg
        with `C_bg` the covariance of the negative samples.

        Note that this reduces to classic PCA when `beta == 0`.

        By choosing `variant == "original"`, the original cPCA approach from Abid et al.
        can be obtained. This amounts to performing PCA on
            (1 - beta) * C_fg - beta * C_bg
        Note the reparametrization compared to Abid et al. Our hyperparameter `beta` is
        related to their `alpha` by `alpha = beta / (1 - beta)`.

        :param n_components: number of top components to keep
        :param beta: (inverse) shrinkage parameter, between 0 and 1; see above
        :param center: whether to center the variables before calculating the covariance
        :param normalize: whether to normalize the variance of each variable to 1 before
            calculating the covariance
        :param variant: which version of cPCA to use; can be
            "original": original version from Abid et al.
            "star":     our variation, cPCA* (see above)
        """
        self.n_components = n_components
        self.beta = beta
        self.center = center
        self.normalize = normalize
        self.variant = variant

        if variant not in ["original", "star"]:
            raise ValueError(f"Unknown variant: {variant}.")

        self.fg_cov_ = None
        self.bg_cov_ = None
        self.bg_cov_shrunk_ = None

        self.eigenvalues_ = None
        self.components_ = None

        self.all_eigenvalues_ = None
        self.all_components_ = None

    def fit(self, X_fg: np.ndarray, X_bg: np.ndarray) -> "OfflineCPCA":
        """Fit the model using the given positive (foreground) and negative (background)
        samples.

        This uses the covariances of the positive (foreground, `X_fg`) and negative
        (background, `X_bg`) samples to calculate the top directions of interest. The
        number of components that are being kept is set by `self.n_components`.

        The following attributes are set:
        `self.fg_cov_`: the covariance of the positive samples
        `self.bg_cov_`: the covariance of the negative samples
        `self.bg_cov_shrunk_`: the shrunk covariance of negative samples,
            (1 - beta) * eye + beta * self.bg_cov_
        `self.eigenvalues_`: the top generalized eigenvalues
        `self.components_`: the top contrastive principle components
        `self.all_eigenvalues_`: all the generalized eigenvalues
        `self.all_components_`: all the generalized eigenvector (contrastive PCs).

        :param X_fg: positive samples; shape `(n_fg, d)`
        :param X_bg: negative samples; shape `(n_bg, d)`
        :return: the cPCA instance, `self`
        """
        X_fg = self._preprocess(X_fg)
        X_bg = self._preprocess(X_bg)

        self.fg_cov_ = X_fg.T @ X_fg / (len(X_fg) - 1)
        self.bg_cov_ = X_bg.T @ X_bg / (len(X_bg) - 1)

        self._solve_eigensystem()

        self.eigenvalues_ = self.all_eigenvalues_[: self.n_components]
        self.components_ = self.all_components_[:, : self.n_components]

        return self

    def update_fit(
        self, beta: Optional[float] = None, n_components: Optional[int] = None
    ) -> "OfflineCPCA":
        """Update the fit while keeping the training data fixed.

        Either the hyperparameter or the number of components (or both) can be updated.

        This takes advantage of the fact that the results of cPCA depend only on the
        covariance structure of the background and foreground samples. This means that
        the covariances do not need to be recalculated when performing cPCA at a
        different hyperparameter value `beta`.

        Similarly, changing the number of components might not require a recalculation.

        :param beta: new value of the hyperparameter
        :param n_components: new number of contrastive PCs to extract
        :return: the cPCA instance, `self`
        """
        if beta is not None and beta != self.beta:
            # need to recalculate the eigensystem
            self.beta = beta
            self._solve_eigensystem()

        if n_components is not None:
            self.n_components = n_components

        self.eigenvalues_ = self.all_eigenvalues_[: self.n_components]
        self.components_ = self.all_components_[:, : self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project a dataset onto the contrastive principal components that were
        identified by a previous call to `fit()`.

        :param X: dataset to transform; shape `(n, d)`
        :return: transformed dataset; shape `(n, self.n_components)`
        """
        return X @ self.components_

    def fit_transform(self, X_fg: np.ndarray, X_bg: np.ndarray) -> np.ndarray:
        """Fit the model and then project the positive samples onto the contrastive PCs.

        :param X_fg: positive samples; shape `(n_fg, d)`
        :param X_bg: negative samples; shape `(n_bg, d)`
        :return: transformed dataset of positive samples; shape `(n, self.n_components)`
        """
        return self.fit(X_fg, X_bg).transform(X_fg)

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Mean-center and normalize the dataset, depending on the instance settings."""
        if self.center:
            X = X - X.mean(axis=0)

        if self.normalize:
            std = X.std(axis=0)
            # avoid division by zero
            std[std == 0] = np.finfo(std.dtype).eps
            X = X / std

        return X

    def _solve_eigensystem(self):
        """Solve for the generalized eigenvalues and eigenvectors given the current
        `fg_cov_`, `bg_cov_`, and `beta`"""
        inv_beta = 1 - self.beta
        if self.variant == "star":
            ident = np.eye(len(self.bg_cov_), dtype=self.bg_cov_.dtype)
            self.bg_cov_shrunk_ = inv_beta * ident + self.beta * self.bg_cov_

            w, vr = scla.eigh(self.fg_cov_, self.bg_cov_shrunk_)
        elif self.variant == "original":
            self.cleaned_cov_ = inv_beta * self.fg_cov_ - self.beta * self.bg_cov_

            w, vr = np.linalg.eigh(self.cleaned_cov_)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        # sort in decreasing order of eigenvalues
        order = np.argsort(w)[::-1]
        w_sorted = w[order]
        vr_sorted = vr[:, order]

        self.all_eigenvalues_ = w_sorted
        self.all_components_ = vr_sorted
