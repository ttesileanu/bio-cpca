import pytest
import numpy as np

from typing import Tuple

from ncpca.models.offline_cpca import OfflineCPCA


@pytest.fixture
def cpca() -> OfflineCPCA:
    return OfflineCPCA()


@pytest.fixture
def rnd_data() -> Tuple[np.ndarray, np.ndarray]:
    n_samples = 10
    d = 5

    rng = np.random.default_rng(143943975)
    X_fg = rng.normal(size=(n_samples, d))
    X_bg = rng.normal(size=(n_samples, d))
    return (X_fg, X_bg)


def test_default_n_components_is_two(cpca):
    assert cpca.n_components == 2


def test_default_beta_is_one_half(cpca):
    assert cpca.beta == 0.5


def test_default_center_is_true(cpca):
    assert cpca.center


def test_default_normalize_is_true(cpca):
    assert cpca.normalize


def test_centering(cpca, rnd_data):
    cpca.center = True
    cpca.fit(*rnd_data)
    res1 = np.copy(cpca.components_)

    rng = np.random.default_rng(42)
    shift0 = rng.normal(size=(rnd_data[0].shape[1],))
    shift1 = rng.normal(size=(rnd_data[1].shape[1],))
    cpca.fit(rnd_data[0] + shift0, rnd_data[1] + shift1)

    np.testing.assert_allclose(res1, cpca.components_)


def test_normalization(cpca, rnd_data):
    cpca.normalize = True
    cpca.fit(*rnd_data)
    res1 = np.copy(cpca.components_)

    rng = np.random.default_rng(42)
    scale0 = 0.5 + rng.uniform(size=(rnd_data[0].shape[1],))
    scale1 = 0.5 + rng.uniform(size=(rnd_data[1].shape[1],))
    cpca.fit(rnd_data[0] * scale0, rnd_data[1] * scale1)

    np.testing.assert_allclose(res1, cpca.components_)


def test_components_size_correct(rnd_data):
    n_components = 3
    cpca = OfflineCPCA(n_components)
    cpca.fit(*rnd_data)

    d = rnd_data[0].shape[1]
    assert cpca.components_.shape == (d, n_components)


@pytest.mark.parametrize("variant", ["original", "star"])
def test_pca_when_beta_is_zero(variant):
    cpca = OfflineCPCA(1, beta=0, normalize=False, variant=variant)

    rng = np.random.default_rng(42)
    a = np.sqrt(3) / 2
    b = np.sqrt(1) / 2
    n = 10_000

    n1 = rng.normal(size=n)
    X_fg1 = np.column_stack((a * n1, b * n1))
    n2 = rng.normal(size=n)
    X_fg2 = 0.1 * np.column_stack((b * n2, -a * n2))

    X_fg = X_fg1 + X_fg2
    X_bg = rng.normal(size=X_fg.shape)
    cpca.fit(X_fg, X_bg)

    ratio = cpca.components_[0, 0] / cpca.components_[1, 0]
    assert pytest.approx(a / b, abs=0.1, rel=0.05) == ratio


def test_X_bg_irrelevant_when_beta_is_zero(cpca, rnd_data):
    cpca.fit(rnd_data[0], rnd_data[1])
    res1 = np.copy(cpca.components_)

    cpca.fit(rnd_data[0], rnd_data[0] - rnd_data[1])

    assert np.mean(np.abs(res1 - cpca.components_)) > 1e-3


def test_cpca_discounts_background_variability():
    cpca = OfflineCPCA(1, beta=0.9, normalize=False)

    rng = np.random.default_rng(42)

    n = 1_000

    # PCA would find (1, 0) as highest variance
    X_fg = np.column_stack((rng.normal(size=n), 0.5 * rng.normal(size=n)))
    # ...but that's high background variance, unlike (0, 1)
    X_bg = np.column_stack((rng.normal(size=n), 0.1 * rng.normal(size=n)))

    cpca.fit(X_fg, X_bg)
    assert np.abs(cpca.components_[0, 0] / cpca.components_[1, 0]) < 0.05


def test_fit_works_with_different_numbers_of_fg_and_bg_samples(cpca):
    d = 5

    rng = np.random.default_rng(34985)
    X_fg = rng.normal(size=(5, d))
    X_bg = rng.normal(size=(25, d))
    cpca.fit(X_fg, X_bg)

    assert cpca.components_.shape == (d, cpca.n_components)


def test_fit_returns_self(cpca, rnd_data):
    res = cpca.fit(*rnd_data)
    assert res is cpca


def test_eigenvalues_shape(cpca, rnd_data):
    cpca.fit(*rnd_data)
    assert cpca.eigenvalues_.shape == (cpca.n_components,)


def test_fit_does_not_change_inputs(cpca, rnd_data):
    rnd_data0 = [np.copy(_) for _ in rnd_data]
    cpca.fit(*rnd_data)

    np.testing.assert_allclose(rnd_data[0], rnd_data0[0])
    np.testing.assert_allclose(rnd_data[1], rnd_data0[1])


def test_fit_transform_uses_components_appropriately(cpca, rnd_data):
    res = cpca.fit_transform(*rnd_data)

    expected = rnd_data[0] @ cpca.components_
    np.testing.assert_allclose(res, expected)


def test_transform_uses_components_appropriately(cpca, rnd_data):
    rng = np.random.default_rng(0)
    new_data = rng.normal(size=(100, rnd_data[0].shape[1]))
    res = cpca.fit(*rnd_data).transform(new_data)

    expected = new_data @ cpca.components_
    np.testing.assert_allclose(res, expected)


def test_update_fit_keeps_fg_and_bg_covs_the_same(cpca, rnd_data):
    cpca.fit(*rnd_data)

    old_fg_cov = np.copy(cpca.fg_cov_)
    old_bg_cov = np.copy(cpca.bg_cov_)

    cpca.update_fit(beta=0.13, n_components=4)

    np.testing.assert_allclose(cpca.fg_cov_, old_fg_cov)
    np.testing.assert_allclose(cpca.bg_cov_, old_bg_cov)


def test_components_shape_correct_after_update_fit(cpca, rnd_data):
    final_n = 3
    assert cpca.n_components != final_n

    cpca.fit(*rnd_data)

    cpca.update_fit(n_components=final_n)

    assert cpca.n_components == final_n
    assert cpca.components_.shape[1] == final_n


def test_results_correct_after_update_fit(cpca, rnd_data):
    cpca.fit(*rnd_data)

    new_beta = 0.213
    new_n = 4
    cpca.update_fit(beta=new_beta, n_components=new_n)

    assert cpca.beta == new_beta

    new_cpca = OfflineCPCA(n_components=new_n, beta=new_beta)
    new_cpca.fit(*rnd_data)

    np.testing.assert_allclose(cpca.components_, new_cpca.components_)
    np.testing.assert_allclose(cpca.eigenvalues_, new_cpca.eigenvalues_)


def test_bg_cov_shrunk_correct_after_update_fit(cpca, rnd_data):
    cpca.fit(*rnd_data)

    new_beta = 0.315
    cpca.update_fit(beta=new_beta)

    eye = np.eye(len(cpca.bg_cov_))
    expected = (1 - new_beta) * eye + new_beta * cpca.bg_cov_

    np.testing.assert_allclose(cpca.bg_cov_shrunk_, expected)


def test_fit_keeps_track_of_all_evals_and_evecs(cpca, rnd_data):
    cpca.fit(*rnd_data)

    n = len(cpca.fg_cov_)
    assert cpca.all_eigenvalues_.shape == (n,)
    assert cpca.all_components_.shape == (n, n)

    d = cpca.n_components
    np.testing.assert_allclose(cpca.all_eigenvalues_[:d], cpca.eigenvalues_)
    np.testing.assert_allclose(cpca.all_components_[:, :d], cpca.components_)


def test_default_variant_is_star(cpca):
    assert cpca.variant == "star"


def test_init_raises_on_unknown_variant():
    with pytest.raises(ValueError):
        OfflineCPCA(variant="blah")


def test_original_variant_pca_of_minus_bg_when_beta_is_one():
    cpca = OfflineCPCA(1, beta=1, normalize=False, variant="original")

    rng = np.random.default_rng(42)
    a = np.sqrt(1) / 2
    b = np.sqrt(3) / 2
    n = 10_000

    n1 = rng.normal(size=n)
    X_bg1 = np.column_stack((a * n1, b * n1))
    n2 = rng.normal(size=n)
    X_bg2 = 0.1 * np.column_stack((b * n2, -a * n2))

    X_bg = X_bg1 + X_bg2

    X_fg = rng.normal(size=X_bg.shape)
    cpca.fit(X_fg, X_bg)

    ratio = -cpca.components_[1, 0] / cpca.components_[0, 0]
    assert pytest.approx(a / b, abs=0.1, rel=0.05) == ratio
