import pytest
import numpy as np

from ncpca.datasets.negative import NegativeGenerator


@pytest.fixture
def positives() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(10, 12, 12))


@pytest.fixture
def negative(positives) -> NegativeGenerator:
    return NegativeGenerator(positives)


def test_sample_returns_correct_size(negative, positives):
    n = 13
    samples = negative.sample(n)

    assert samples.shape == (n, *positives.shape[1:])


def test_use_custom_generator(positives):
    n = 3

    seed = 21
    rng = np.random.default_rng(21)
    negative = NegativeGenerator(positives, rng=rng)
    samples1 = negative.sample(n)

    rng = np.random.default_rng(21)
    negative = NegativeGenerator(positives, rng=rng)
    samples2 = negative.sample(n)

    np.testing.assert_allclose(samples1, samples2)


def test_set_blur_iter(positives):
    seed = 42
    negative1 = NegativeGenerator(positives, rng=seed, blur_iter=4)

    n = 3

    samples1 = negative1.sample(n)

    negative2 = NegativeGenerator(positives, rng=seed, blur_iter=2)
    samples2 = negative2.sample(n)

    assert np.mean(np.abs(samples1 - samples2)) > 1e-3


def test_return_parts(negative, positives):
    n = 13
    samples = negative.sample(n, return_parts=True)

    assert samples.shape == (4, n, *positives.shape[1:])


def test_output_is_random(negative):
    n = 13
    samples1 = negative.sample(n)
    samples2 = negative.sample(n)

    assert np.mean(np.abs(samples1 - samples2)) > 1e-3


def test_mask_type_half():
    positives = np.stack(
        [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ]
    )
    negative = NegativeGenerator(positives, mask_type="half")
    sample = negative.sample(1).squeeze()

    possible1 = np.array([[1.0, 2.0], [7.0, 8.0]])
    possible2 = np.array([[5.0, 6.0], [3.0, 4.0]])

    assert np.allclose(sample, possible1) or np.allclose(sample, possible2)


def test_mask_type_half_nontrivial_jitter():
    positives = np.stack(
        [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ]
    )
    negative = NegativeGenerator(positives, mask_type="half", cut_jitter=1)
    sample = negative.sample(50)

    found_originals = [False, False]
    for crt in sample:
        if np.allclose(crt, positives[0]):
            found_originals[0] = True
        if np.allclose(crt, positives[1]):
            found_originals[1] = True

    assert all(found_originals)
