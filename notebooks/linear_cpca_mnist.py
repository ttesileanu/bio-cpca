# %% [markdown]
# # Test linear cPCA on MNIST

import torch
import torchvision
import torchvision.transforms as T

import numpy as np

import matplotlib.pyplot as plt
import pydove as dv
import seaborn as sns

from tqdm.auto import tqdm

from sklearn.neural_network import MLPClassifier

from ncpca import NegativeGenerator, OfflineCPCA

# %% [markdown]
# # Experiment with contrastive methods
# ## Generate data sets

# %%
torch.manual_seed(0)

transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(
    "data/", train=True, transform=transform, download=True
)

n_samples = 10_000
positives = np.concatenate([trainset[_][0].double().numpy() for _ in range(n_samples)])
positive_labels = [trainset[_][1] for _ in range(n_samples)]

negative_source = np.concatenate(
    [trainset[_][0].double().numpy() for _ in range(n_samples, 2 * n_samples)]
)
negative_generator = NegativeGenerator(negative_source, blur_iter=8)
# negative_generator = NegativeGenerator(negative_source, mask_type="half")
negatives_detailed = negative_generator.sample(n_samples, return_parts=True)
negatives = negatives_detailed[0]

# %% [markdown]
# Some example positive and negative samples.

# %%
with dv.FigureManager(4, 4, figsize=(10, 10), do_despine=False) as (_, axs):
    for i in range(8):
        axs[i // 4, i % 4].imshow(positives[i], cmap="gray")
        axs[2 + i // 4, i % 4].imshow(negatives[i], cmap="gray")

    for i in range(2):
        axs[i, 0].set_ylabel("positive")
        axs[2 + i, 0].set_ylabel("negative")

    for i in range(4):
        for j in range(4):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

# %% [markdown]
# Details of negative samples.

# %%
n_shown = 5
with dv.FigureManager(n_shown, 4, figsize=(10, 2.5 * n_shown), do_despine=False) as (
    _,
    axs,
):
    for i in range(n_shown):
        axs[i, 0].imshow(negatives_detailed[1, i], cmap="gray")
        axs[i, 1].imshow(negatives_detailed[3, i], cmap="gray")
        axs[i, 2].imshow(negatives_detailed[2, i], cmap="gray")
        axs[i, 3].imshow(negatives_detailed[0, i], cmap="gray")

        for j in range(4):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    axs[0, 0].set_title("img 1")
    axs[0, 1].set_title("mask")
    axs[0, 2].set_title("img 2")
    axs[0, 3].set_title("result")

# %% [markdown]
# ## Check what PCA does.

# %%
d = 100
positives_vec = positives.reshape(len(positives), -1)
negatives_vec = negatives.reshape(len(negatives), -1)

pca = OfflineCPCA(d, beta=0.0)
positives_pca = pca.fit_transform(positives_vec, negatives_vec)

# %%
with dv.FigureManager() as (_, ax):
    sns.scatterplot(
        x=positives_pca[:, 0],
        y=positives_pca[:, 1],
        size=2,
        alpha=0.5,
        hue=positive_labels,
        ax=ax,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

# %% [markdown]
# Try cPCA*.

# %%
cpca = OfflineCPCA(d, beta=0.5)
positives_cpca = cpca.fit_transform(positives_vec, negatives_vec)

# %%
with dv.FigureManager() as (_, ax):
    sns.scatterplot(
        x=positives_cpca[:, 0],
        y=positives_cpca[:, 1],
        size=2,
        alpha=0.5,
        hue=positive_labels,
        ax=ax,
    )
    ax.set_xlabel("cPC1")
    ax.set_ylabel("cPC2")

# %% [markdown]
# ## Train linear classifiers on PCA and cPCA* outputs
pca_classifier = MLPClassifier(
    hidden_layer_sizes=tuple(), max_iter=1000, random_state=42, activation="identity"
)
pca_classifier.fit(positives_pca[: n_samples // 2], positive_labels[: n_samples // 2])

pca_score = pca_classifier.score(
    positives_pca[n_samples // 2 :], positive_labels[n_samples // 2 :]
)
print(f"PCA test set score: {pca_score}.")

cpca_classifier = MLPClassifier(
    hidden_layer_sizes=tuple(), max_iter=1000, random_state=42, activation="identity"
)
cpca_classifier.fit(positives_cpca[: n_samples // 2], positive_labels[: n_samples // 2])

cpca_score = cpca_classifier.score(
    positives_cpca[n_samples // 2 :], positive_labels[n_samples // 2 :]
)
print(f"cPCA* test set score: {cpca_score}.")

# %% [markdown]
# ## Visualize the (contrastive) principal components

# %%
with dv.FigureManager(2, 5, figsize=(10, 4)) as (fig, axs):
    for i in range(min(d, 10)):
        ax = axs[i // 5, i % 5]
        crt_mode = pca.components_[:, i].reshape(28, 28)
        crt_max = np.max(np.abs(crt_mode))
        ax.imshow(crt_mode, vmin=-crt_max, vmax=crt_max, cmap="RdBu")

    fig.suptitle("PCs")

# %%
with dv.FigureManager(2, 5, figsize=(10, 4)) as (fig, axs):
    for i in range(min(d, 10)):
        ax = axs[i // 5, i % 5]
        crt_mode = cpca.components_[:, i].reshape(28, 28)
        crt_max = np.max(np.abs(crt_mode))
        ax.imshow(crt_mode, vmin=-crt_max, vmax=crt_max, cmap="RdBu")

    fig.suptitle("cPCs")

# %%
with dv.FigureManager(2, 5, figsize=(10, 4)) as (fig, axs):
    tmp = np.linalg.eigh(pca.bg_cov_)
    tmp_order = np.flip(np.argsort(tmp[0]), (0,))
    tmp_evecs = tmp[1][:, tmp_order]
    for i in range(min(d, 10)):
        ax = axs[i // 5, i % 5]
        crt_mode = tmp_evecs[:, i].reshape(28, 28)
        crt_max = np.max(np.abs(crt_mode))
        ax.imshow(crt_mode, vmin=-crt_max, vmax=crt_max, cmap="RdBu")

    fig.suptitle("negative PCs")

# %%
with dv.FigureManager(1, 2, do_despine=False) as (fig, (ax1, ax2)):
    crt_max = max(np.max(np.abs(pca.fg_cov_)), np.max(np.abs(pca.bg_cov_)))

    ax1.imshow(pca.fg_cov_, vmin=-crt_max, vmax=crt_max, cmap="RdBu")
    ax1.set_title("positive samples")

    ax2.imshow(pca.bg_cov_, vmin=-crt_max, vmax=crt_max, cmap="RdBu")
    ax2.set_title("negative samples")

# %%
with dv.FigureManager() as (_, ax):
    ax.scatter(pca.fg_cov_.ravel(), pca.bg_cov_.ravel(), s=1, alpha=0.02)
    ax.set_xlabel("cov(positives)")
    ax.set_xlabel("cov(negatives)")

    ax.plot(ax.get_xlim(), ax.get_xlim(), "k--", lw=1, label="$y=x$")
    ax.plot(
        ax.get_xlim(), 0.5 * np.asarray(ax.get_xlim()), "r--", lw=1, label="$y=x/2$"
    )

    ax.legend(frameon=False)

# %%
negatives_pca = pca.transform(negatives_vec)
negatives_cpca = cpca.transform(negatives_vec)
with dv.FigureManager(1, 2) as (_, (ax1, ax2)):
    ax1.scatter(
        positives_pca[:, 0], positives_pca[:, 1], s=1.5, alpha=0.4, label="positives"
    )
    ax1.scatter(
        negatives_pca[:, 0], negatives_pca[:, 1], s=1.5, alpha=0.4, label="negatives"
    )
    ax1.legend(frameon=False)

    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")

    ax1.set_title("PCA on negatives")

    ax2.scatter(
        positives_cpca[:, 0], positives_cpca[:, 1], s=1.5, alpha=0.4, label="positives"
    )
    ax2.scatter(
        negatives_cpca[:, 0], negatives_cpca[:, 1], s=1.5, alpha=0.4, label="negatives"
    )
    ax2.legend(frameon=False)

    ax2.set_xlabel("cPC1")
    ax2.set_ylabel("cPC2")

    ax2.set_title("cPCA* on negatives")

# %% [markdown]
# ## Sweep over parameters

# %%
d_values = np.geomspace(2, 784, 10).astype(int)
beta_values = np.linspace(0, 0.99, 10)

classifier_accuracy = np.zeros((len(d_values), len(beta_values)))

sweep_cpca = OfflineCPCA(d_values[0], beta=beta_values[0])
sweep_cpca.fit(positives_vec, negatives_vec)
for j, beta in enumerate(tqdm(beta_values, desc="beta")):
    for i, d in enumerate(tqdm(d_values, desc="dimension")):
        sweep_cpca.update_fit(beta=beta, n_components=d)
        sweep_positives = sweep_cpca.transform(positives_vec)

        sweep_classifier = MLPClassifier(
            hidden_layer_sizes=tuple(),
            max_iter=1000,
            random_state=42,
            activation="identity",
        )
        sweep_classifier.fit(
            sweep_positives[: n_samples // 2], positive_labels[: n_samples // 2]
        )

        classifier_accuracy[i, j] = sweep_classifier.score(
            sweep_positives[n_samples // 2 :], positive_labels[n_samples // 2 :]
        )

# %% [markdown]
# Visualize the sweep.

# %%
with dv.FigureManager() as (_, ax):
    # h = ax.imshow(classifier_accuracy)
    h = sns.heatmap(
        data=classifier_accuracy, xticklabels=beta_values, yticklabels=d_values, ax=ax
    )
    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("number of cPCs")

# %%
