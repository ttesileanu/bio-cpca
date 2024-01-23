# %% [markdown]
# # Compare cPCA to cPCA* on corrupted MNIST
#! %load_ext autoreload
#! %autoreload 2

import os
from tqdm import tqdm

import torchvision
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

from scipy.io import loadmat
from sklearn.decomposition import PCA

import PIL

# import pickle

from ncpca import OfflineCPCA, resize_and_crop, cloud_distance

# %% [markdown]
# ## Load MNIST

# %%
transform = T.Compose([T.ToTensor()])
trainset = torchvision.datasets.MNIST(
    "data/", train=True, transform=transform, download=True
)

train_images = np.concatenate(
    [np.reshape(_[0].double().numpy(), (1, -1)) for _ in trainset]
)
train_labels = np.array([_[1] for _ in trainset])

target_idx = np.where(train_labels < 2)[0]
foreground = train_images[target_idx, :][:5000]
target_labels = train_labels[target_idx][:5000]

# %% [markdown]
# ## PCA on Regular MNIST

# %%
pca = PCA(n_components=2)
fg = pca.fit_transform(foreground)
colors = ["k", "r"]

with dv.FigureManager() as (_, ax):
    for i, l in enumerate(np.sort(np.unique(target_labels))):
        ax.scatter(
            fg[np.where(target_labels == l), 0],
            fg[np.where(target_labels == l), 1],
            color=colors[i],
            label="Digit " + str(l),
            alpha=0.5,
        )
    ax.legend(frameon=False)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA on uncorrupted MNIST")

# %% [markdown]
# ## Load natural images
# *The original link from Abid et al. (see below) does not work anymore, and I can't
# find the 'grass' synset. I used images from the UPenn Natural Image Database instead.*
#
# **OLD:** These pictures are found in this [OneDrive link](https://1drv.ms/f/s!AgLi37o1j88ahrJTLeycjuEoHpVhQw), or they can be downloaded from [ImageNet](http://image-net.org/download) using the synset 'grass'. (*Note*: replace IMAGE_PATH with path to the downloaded images)

# %%
IMAGE_PATH = os.path.join("data", "backgrounds")

natural_images = (
    list()
)  # dictionary of pictures indexed by the pic # and each value is 100x100 image
for subdir in tqdm(os.listdir(IMAGE_PATH)):
    if subdir.startswith("."):
        continue
    crt_path = os.path.join(IMAGE_PATH, subdir)
    for filename in os.listdir(crt_path):
        if filename.endswith("LUM.mat") or filename.endswith("LUM.MAT"):
            im = loadmat(os.path.join(crt_path, filename))
            im = im["LUM_Image"]
            im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255.0
            im = PIL.Image.fromarray(np.uint8(im))
            im = resize_and_crop(
                im
            )  # resize and crop each picture to be 100px by 100px
            natural_images.append(np.reshape(im, [10000]))

natural_images = np.asarray(natural_images, dtype=float)
natural_images /= 255  # rescale to be 0-1
print("Array of grass images:", natural_images.shape)

# %% [markdown]
# ## Corrupt MNIST by Superimposing Images of Nature
# To create each of the 5000 corrupted digits, randomly choose a 28px by 28px region
# from a nature image and superimpose on top of the digits.

# %%
rng = np.random.default_rng(42)

rand_indices = rng.permutation(natural_images.shape[0])  # just shuffles the indices
split = int(len(rand_indices) / 2)
target_indices = rand_indices[
    0:split
]  # choose the first half of images to be superimposed on target
background_indices = rand_indices[
    split:
]  # choose the second half of images to be background dataset

target = np.zeros(foreground.shape)
background = np.zeros(foreground.shape)

for i in range(target.shape[0]):
    idx = rng.choice(target_indices)  # randomly pick a image
    loc = rng.integers(70, size=(2))  # randomly pick a region in the image
    superimposed_patch = np.reshape(
        np.reshape(natural_images[idx, :], [100, 100])[loc[0] : loc[0] + 28, :][
            :, loc[1] : loc[1] + 28
        ],
        [1, 784],
    )
    target[i] = 0.25 * foreground[i] + superimposed_patch

    idx = rng.choice(background_indices)  # randomly pick a image
    loc = rng.integers(70, size=(2))  # randomly pick a region in the image
    background_patch = np.reshape(
        np.reshape(natural_images[idx, :], [100, 100])[loc[0] : loc[0] + 28, :][
            :, loc[1] : loc[1] + 28
        ],
        [1, 784],
    )
    background[i] = background_patch

# %% [markdown]
# ## Some Example Images

# %%
n_show = 6

plt.figure(figsize=[12, 12])
for i in range(n_show):
    plt.subplot(1, n_show, i + 1)
    idx = np.random.randint(5000)
    plt.imshow(
        np.reshape(target[idx, :], [28, 28]), cmap="gray", interpolation="bicubic"
    )
    plt.axis("off")

plt.figure(figsize=[12, 12])
for i in range(n_show):
    plt.subplot(1, n_show, i + 1)
    idx = np.random.randint(5000)
    plt.imshow(
        np.reshape(background[idx, :], [28, 28]), cmap="gray", interpolation="bicubic"
    )
    plt.axis("off")


# %% [markdown]
# ## PCA on Corrupted MNIST

# %%
pca = PCA(n_components=2)
fg = pca.fit_transform(target)

with dv.FigureManager() as (_, ax):
    for i, l in enumerate(np.sort(np.unique(target_labels))):
        ax.scatter(
            fg[np.where(target_labels == l), 0],
            fg[np.where(target_labels == l), 1],
            color=colors[i],
            label="Digit " + str(l),
            alpha=0.5,
        )
    ax.legend(frameon=False)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA on corrupted MNIST")

# %% [markdown]
# ## Perform Contrastive PCA

# %%
cpca_original = OfflineCPCA(n_components=2, beta=0, variant="original")

beta_values = np.linspace(0, 0.99, 100)
cpca_original.fit(target, background)
projections_original = []
for beta in tqdm(beta_values):
    cpca_original.update_fit(beta=beta)
    res = cpca_original.transform(target)
    projections_original.append(res)

# %%

n_plots = 5

with dv.FigureManager(
    1, n_plots, figsize=(2 * n_plots, 2), despine_kws={"offset": 0}
) as (fig, axs):
    plot_idxs = list(np.linspace(0, len(beta_values) - 1, n_plots).astype(int))

    color_map = ["k", "r", "b", "g", "c"]
    dist_values = np.zeros(len(beta_values))
    for i in range(len(beta_values)):
        crt_proj = projections_original[i]
        crt_proj_by_class = []
        for cls in range(2):
            mask = target_labels == cls
            crt_proj_by_class.append(crt_proj[mask, :])

            if i in plot_idxs:
                ax = axs[plot_idxs.index(i)]
                ax.scatter(
                    crt_proj[mask, 0],
                    crt_proj[mask, 1],
                    c=color_map[cls],
                    label=f"Class {cls}",
                    alpha=0.6,
                )

        crt_dist = cloud_distance(crt_proj_by_class[0], crt_proj_by_class[1])
        dist_values[i] = crt_dist

        if i in plot_idxs:
            ax.set_title(f"α={beta_values[i]:.1f}, D={crt_dist:.3g}", fontsize=10)

    axs[-1].legend()
    fig.suptitle("cPCA original")

# %% [markdown]
# ## Peform cPCA*

# %%
cpca_star = OfflineCPCA(n_components=2, beta=0, variant="star")

projections_star = []
cpca_star.fit(target, background)
for beta in tqdm(beta_values):
    cpca_star.update_fit(beta=beta)
    res_star = cpca_star.transform(target)
    projections_star.append(res_star)

# %%

n_plots = 5
with dv.FigureManager(
    1, n_plots, figsize=(2 * n_plots, 2), despine_kws={"offset": 0}
) as (fig, axs):
    plot_idxs = list(np.linspace(0, len(beta_values) - 1, n_plots).astype(int))
    dist_values_div = np.zeros(len(beta_values))
    for i in range(len(beta_values)):
        crt_proj = projections_star[i]
        crt_proj_by_class = []
        for cls in range(2):
            mask = target_labels == cls
            crt_proj_by_class.append(crt_proj[mask, :])

            if i in plot_idxs:
                ax = axs[plot_idxs.index(i)]
                ax.scatter(
                    crt_proj[mask, 0],
                    crt_proj[mask, 1],
                    c=color_map[cls],
                    label=f"Class {cls}",
                    alpha=0.6,
                )

        crt_dist = cloud_distance(crt_proj_by_class[0], crt_proj_by_class[1])
        dist_values_div[i] = crt_dist

        if i in plot_idxs:
            ax.set_title(f"α={beta_values[i]:.1g}, D={crt_dist:.3g}", fontsize=10)

    axs[-1].legend()
    fig.suptitle("cPCA*")

# %%
perfect_pca = OfflineCPCA(n_components=2, beta=0, variant="original")
# `background`` is not actually used here
perfect_pca_proj = perfect_pca.fit_transform(foreground, background)
perfect_dist = cloud_distance(
    perfect_pca_proj[target_labels == 0], perfect_pca_proj[target_labels == 1]
)
print(f"Foreground D={perfect_dist:.4g}.")

# %%
with dv.FigureManager() as (_, ax):
    ax.plot(beta_values, dist_values, c="C0", label="original")
    ax.set_xlabel("alpha")
    ax.set_ylabel("distance")

    ax.plot(beta_values, dist_values_div, c="C1", label="division")
    ax.legend(frameon=False)

# %% [markdown]
# ## Make plots for poster
# ### Show PCA and cPCA* on the corrupted digits

# %%
with dv.FigureManager(
    figsize=(2.5, 1.71), despine_kws=dict(left=True, bottom=True)
) as (fig, ax):
    ax.axhline(0, ls="--", lw=1, c="#4D4E4C", zorder=-1)
    ax.axvline(0, ls="--", lw=1, c="#4D4E4C", zorder=-1)

    mask = target_labels == 0
    centered = projections_star[0] - np.mean(projections_star[0], axis=0)
    ax.scatter(centered[mask, 0], centered[mask, 1], alpha=0.3, c="#0076BA", s=2)
    ax.scatter(centered[~mask, 0], centered[~mask, 1], alpha=0.3, c="#F27200", s=2)

    # xl = np.max(np.abs(ax.get_xlim()))
    # yl = np.max(np.abs(ax.get_ylim()))
    xl = np.quantile(np.abs(centered[:, 0]), 0.99)
    yl = np.quantile(np.abs(centered[:, 1]), 0.99)
    ax.set_xlim(-xl, xl)
    ax.set_ylim(-yl, yl)

    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig(os.path.join("..", "figs", "poster", "pca_corrupt_mnist.png"), dpi=300)

# %%
with dv.FigureManager(
    figsize=(2.5, 1.71), despine_kws=dict(left=True, bottom=True)
) as (fig, ax):
    ax.axhline(0, ls="--", lw=1, c="#4D4E4C", zorder=-1)
    ax.axvline(0, ls="--", lw=1, c="#4D4E4C", zorder=-1)

    mask = target_labels == 0
    centered = projections_star[-1] - np.mean(projections_star[-1], axis=0)
    ax.scatter(centered[mask, 0], centered[mask, 1], alpha=0.3, c="#0076BA", s=2)
    ax.scatter(centered[~mask, 0], centered[~mask, 1], alpha=0.3, c="#F27200", s=2)

    # xl = np.max(np.abs(ax.get_xlim()))
    # yl = np.max(np.abs(ax.get_ylim()))
    xl = np.quantile(np.abs(centered[:, 0]), 0.99) * 1.4
    yl = np.quantile(np.abs(centered[:, 1]), 0.99) * 1.4
    ax.set_xlim(-xl, xl)
    ax.set_ylim(-yl, yl)

    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig(
    os.path.join("..", "figs", "poster", "cpca_star_corrupt_mnist.png"), dpi=300
)

# %% [markdown]
# ## Show some corrupted digits

# %%
for digit in range(2):
    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_axes([0, 0, 1, 1])
    idx = np.where(target_labels == digit)[0][0]
    ax.imshow(np.reshape(target[idx, :], (28, 28)), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(
        os.path.join("..", "figs", "poster", f"corrupted_digit_{digit}.png"), dpi=300
    )

# %% [markdown]
# ## Show separation vs. hyperparameter

# %%
# %%
with dv.FigureManager(figsize=(4.5, 2.56), offset=5) as (fig, ax):
    ax.plot(beta_values, dist_values_div, c="#0076BA", lw=3)
    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("cluster distance*")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, None)

fig.savefig(os.path.join("..", "figs", "poster", "cpca_star_vs_beta.pdf"))

# %% [markdown]
# ## Test convergence of iterative solution
# Offline first.

# %%

beta = 0.9
Z_dim = cpca_star.n_components
eta_it = 1e-3
epochs = 10_000
# runs = 10
runs = 2
tau = 0.5
U0 = 1.0

cpca_star.update_fit(beta=beta)
X = cpca_star._preprocess(target).T
Y = cpca_star._preprocess(background).T

num_samples = X.shape[1]

eigvec = cpca_star.components_.T
eigvec2 = eigvec @ eigvec.T
proj_eigvec = eigvec.T @ np.linalg.inv(eigvec2) @ eigvec

rng = np.random.default_rng(42)
V_it = rng.normal(size=(Z_dim, X.shape[0]))
U_it = U0 * np.eye(Z_dim, Z_dim)
# T = 2 * num_samples
T = num_samples

dU_history = np.zeros(epochs)
dV_history = np.zeros(epochs)
overlap_history = np.zeros(epochs)

for e in tqdm(range(epochs)):
    Z = np.linalg.inv(U_it) @ V_it @ X
    dV = 2 * eta_it * ((1 / T) * Z @ X.T - V_it @ cpca_star.bg_cov_shrunk_)
    dU = (eta_it / tau) * ((1 / T) * Z @ Z.T - U_it)

    eff_V = V_it
    VVT = eff_V @ eff_V.T
    projV_it = eff_V.T @ np.linalg.inv(VVT) @ eff_V

    V_det = np.linalg.det(VVT)
    V_span = np.trace(projV_it)
    V_overlap = np.abs(np.trace(proj_eigvec @ projV_it))

    dU_history[e] = np.max(np.abs(dU))
    dV_history[e] = np.max(np.abs(dV))
    overlap_history[e] = V_overlap

    U_it += dU
    V_it += dV

# %%

with dv.FigureManager(1, 3) as (_, (ax1, ax2, ax3)):
    ax1.semilogy(dU_history)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("max(abs(dU))")

    ax2.semilogy(dV_history)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("max(abs(dV))")

    ax3.plot(overlap_history, "--k")
    ax3.axhline(Z_dim)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("subspace alignment")


# %% [markdown]
# Now online.

# %%
# adapted from code from Siavash
# eta_weight = 0.0004
eta_weight = 3e-5
num_reports = 50
epochs = 60
runs = 25
tau = 0.2
# runs = 2

rng = np.random.default_rng(42)
progress_step = max([1, (2 * num_samples * epochs) // num_reports])

overlap_hists = []
for run in tqdm(range(runs), desc="run"):
    V = rng.normal(size=(Z_dim, X.shape[0]))
    U = U0 * np.eye(Z_dim, Z_dim)

    loss_hist = []
    overlap_hist = []
    sample_hist = []

    counter = 0
    for e in tqdm(range(epochs), desc="epoch"):
        for t in range(2 * num_samples):
            if t % 2 == 0:
                x = X[:, t // 2]
                y = 1
            else:
                x = Y[:, t // 2]
                y = 0

            c = V @ x
            Z = y * np.linalg.inv(U) @ c

            Vnew = V + 2 * eta_weight * (
                (Z - 2 * beta * (1 - y) * c).reshape(-1, 1).dot(x.reshape(1, -1))
                - (1 - beta) * V
            )

            # if t%2:
            Unew = U + eta_weight / tau * (
                Z.reshape(-1, 1).dot(Z.transpose().reshape(1, -1)) - U
            )
            # else:
            #     # Unew = U + 1*eta_weight*( Z.reshape(-1,1).dot(N.transpose().reshape(1,-1)) - U )
            #     Unew = U

            if counter % progress_step == 0:
                # projV = V.transpose().dot(np.linalg.inv(V.dot(V.transpose())).dot(V))
                VVT = V @ V.T
                projV = V.T @ np.linalg.inv(VVT) @ V

                dV = np.abs(Vnew - V).max()
                V_det = np.linalg.det(VVT)
                V_span = np.trace(projV)
                V_overlap = np.abs(np.trace(proj_eigvec @ projV))

                # print("Sample {}:".format(counter))
                # # print('Final N gradient magnitude: {:.2g}, Final Z gradient magnitude: {:.2g}'.format(N_grad.dot(N_grad),Z_grad.dot(Z_grad)))
                # print(
                #     f"V update: {dV:.2g}, V det: {V_det:.2g}, V span {V_span:.2g}, "
                #     f"V overlap: {V_overlap:.3g}"
                # )

                # print()

                overlap_hist.append(V_overlap)

                sample_hist.append(counter)

            V = Vnew
            U = Unew
            counter += 1

    print(overlap_hist[-1])
    overlap_hists.append(np.array(overlap_hist))

overlap_hists = np.array(overlap_hists)

# %%
with dv.FigureManager(figsize=(3, 1.7), offset=5) as (fig, ax):
    blue = "#0076BA"
    ax.plot(
        sample_hist,
        np.median(overlap_hists, axis=0),
        c=blue,
        label="Online cPCA$^\\ast$",
    )
    ax.fill_between(
        sample_hist,
        np.quantile(overlap_hists, 0.16, axis=0),
        np.quantile(overlap_hists, 0.84, axis=0),
        color=blue,
        edgecolor="none",
        alpha=0.5,
    )
    ax.axhline(Z_dim, c="k", dashes=[3, 3], lw=1, label="Optimal projection")
    ax.set_ylabel("subspace alignment", fontsize=7)
    ax.set_xlabel("samples", fontsize=7)

    ax.tick_params(axis="both", labelsize=7)

    x_ticks = np.arange(0, 500_001, 100_000)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{int(np.round(_ / 1000))}k" for _ in x_ticks])

    ax.set_xlim(0, max(sample_hist))

    ax.legend(frameon=False, fontsize=8)

fig.savefig(os.path.join("..", "figs", "poster", "corrupted_mnist_overlap.pdf"))

# %%

# results = {
#     "alpha_values": beta_values,
#     "alpha_orig_values": alpha_orig_values,
#     "dist_values_orig": dist_values,
#     "dist_values_div": dist_values_div,
#     "projections_orig": projections_original,
#     "projections_div": projections_star,
#     "labels": target_labels,
#     "target": target,
#     "foreground": foreground,
#     "background": background,
# }

# with open(os.path.join("..", "save", "comparison_mnist.pkl"), "wb") as f:
#     pickle.dump(results, f)

# %%
