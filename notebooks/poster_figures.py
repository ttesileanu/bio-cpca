# %% [markdown]
# # Making some figures for the Cosyne poster.
import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

import numpy as np

import os

# %% [markdown]
# ## Example where PCA fails.

# %%
rng = np.random.default_rng(42)

n = 5_000
raw_signal = 2.0 * rng.integers(0, 2, size=n) - 1

v_signal = np.array([0, 1])
v_noise = np.array([1, 0])

signal = raw_signal[:, None] * v_signal[None, :]

noise = rng.normal(scale=[1.5, 0.1], size=(n, 2))

total = signal + noise

with dv.FigureManager(figsize=(6, 2), despine_kws={"left": True, "bottom": True}) as (
    fig,
    ax,
):
    mask = raw_signal == 1
    ax.scatter(total[mask, 0], total[mask, 1], alpha=0.2, c="#0076BA")
    ax.scatter(total[~mask, 0], total[~mask, 1], alpha=0.2, c="#F27200")

    cov = np.cov(total.T)
    pc1 = np.linalg.eigh(cov)[1][:, 1]
    pc1 *= np.sign(pc1[0])

    # ax.plot([0, pc1[0]], [0, pc1[1]])
    ax.arrow(
        0,
        0,
        *(2 * pc1),
        lw=3,
        head_length=0.1,
        head_width=0.1,
        fc="#C0504D",
        ec="#C0504D",
    )

    ax.set_aspect(1)

    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig(
    os.path.join("..", "figs", "poster", "pca_fail.png"), dpi=300, transparent=True
)

# %%
