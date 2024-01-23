import numpy as np
from scipy.signal import convolve2d

from typing import Sequence, Tuple, Union


class NegativeGenerator:
    """Generate negative (out-of-distribution) samples by randomly mixing pairs of
    positive (in-distribution) samples.

    This focuses on the case of images. Masks are by default ("blobs") generated
    starting with a random 0-1 pattern, blurring it repeatedly, then thresholding at 0.5
    (as in Hinton's FF paper).

    Other options for generating masks are also available.
    """

    def __init__(
        self,
        positives: Sequence[np.ndarray],
        mask_type: str = "blobs",
        rng: Union[None, int, np.random.Generator] = 0,
        blur_iter: int = 4,
        cut_jitter: int = 0,
    ):
        """Initialize.

        :param positives: set of positive samples to use for generating the negative
            samples
        :param mask_type: type of masks to use; can be
            "blobs":    Hinton-style blobs generated from random 0-1 patterns
            "half":     choose top of one sample, bottom from other, with small jitter
                        in position of half-line (see `cut_jitter`)
        :param rng: random number generator to use
        :param blur_iter: number of blurring iterations to use when generating masks
        :param cut_jitter: amount of random jitter (in pixels), in either direction, in
            the separation line between top and bottom halves when `mask_type == "half"`
        """
        self.positives = positives
        self.mask_type = mask_type
        self.rng = np.random.default_rng(rng)
        self.blur_iter = blur_iter
        self.cut_jitter = cut_jitter

        self.kernel = np.array(
            [
                [1 / 16, 1 / 8, 1 / 16],
                [1 / 8, 1 / 4, 1 / 8],
                [1 / 16, 1 / 8, 1 / 16],
            ]
        )

    def generate_mask(self, n_rows: int, n_cols: int) -> np.ndarray:
        if self.mask_type == "blobs":
            mask = self.rng.integers(0, 2, size=(n_rows, n_cols)).astype(float)
            for i in range(self.blur_iter):
                mask = convolve2d(mask, self.kernel, mode="same")

            mask = (mask > 0.5).astype(float)
        else:
            mask = np.zeros((n_rows, n_cols))
            if self.cut_jitter > 0:
                jitter = self.rng.integers(0, self.cut_jitter + 1).item()
                sep = min(max(n_rows // 2 + jitter, 0), n_rows)
            else:
                sep = n_rows // 2

            mask[sep:, :] = 1.0

        return mask.squeeze()

    def generate_hybrid(
        self, im1: np.ndarray, im2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert im1.ndim == 2
        assert im2.ndim == 2

        assert im1.shape == im2.shape

        mask = self.generate_mask(*im1.shape)
        return im1 * mask + im2 * (1 - mask), mask

    def sample(self, n: int, return_parts: bool = False) -> np.ndarray:
        """Generate samples.

        :param n: number of samples to generate
        :param return_parts: if true, the input samples as well as the masks used to
            generate the outputs are also returned; see below
        :return: a tensor of size `(n, n_rows, n_cols)` if `return_parts` is False;
            otherwise a tensor of size `(4, n, n_rows, n_cols)`, where the second
            and third element contain the two input samples that were combined, and the
            last element contains the mask that was used
        """
        assert len(self.positives) >= 2
        idx1 = self.rng.integers(0, len(self.positives), size=n)

        # ensure idx2 is always != idx1
        idx2 = self.rng.integers(0, len(self.positives) - 1, size=n)
        idx2[idx2 >= idx1] += 1

        shape = self.positives.shape[1:]
        if not return_parts:
            result = np.empty((n, *shape))
        else:
            result = np.empty((4, n, *shape))
        for i in range(n):
            im1 = self.positives[idx1[i]]
            im2 = self.positives[idx2[i]]
            crt_result, crt_mask = self.generate_hybrid(im1, im2)
            if not return_parts:
                result[i] = crt_result
            else:
                result[0, i] = crt_result
                result[1, i] = im1
                result[2, i] = im2
                result[3, i] = crt_mask

        return result
