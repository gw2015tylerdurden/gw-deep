import os
import hydra
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as tf
#from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from tqdm import tqdm
from sklearn.manifold import TSNE

import src.data.datasets as datasets
import src.nn.models as models
import src.utils.transforms as transforms
import src.utils.functional as F
import src.utils.logging as logging
from sklearn.metrics import silhouette_samples
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
    
plt.style.use("seaborn-poster")
plt.rcParams["lines.markersize"] = 6.0
plt.rc("legend", fontsize=10)

def plotReconsructImages(input, output, save_figure_path, numplots=4):

    # output channle of image is 0[0.5sec]
    channel = 0
    # shuffle and get first numplots indices of input
    indices = torch.randperm(len(input))[:numplots]
    x_random_sample = input[indices].cpu().numpy()
    x_rec_random_sample = output[indices].cpu().numpy()

    # numplots input and output, i.e. numplots *=2
    fig, ax = plt.subplots(nrows=2, ncols=numplots, figsize=(numplots*1.5, numplots*1.5), sharex=True, sharey=True, tight_layout=True)
    for sample in range(numplots):
        x_sample = x_random_sample[sample]
        x_rec_sample = x_rec_random_sample[sample]

        # [2*sample, xxx] is for input, [2*sample+1, xxx] is for output
        x = x_sample[channel]
        x_rec = x_rec_sample[channel]

        # plots row 2-images
        ax1 = ax[0, sample]
        ax2 = ax[1, sample]

        # settigns for visualization
        ax1.axis("off")
        ax2.axis("off")

        # plot gray scale
        ax1.imshow(x, cmap='gray', vmin=0, vmax=1, interpolation='none')
        ax2.imshow(x_rec, cmap='gray', vmin=0, vmax=1, interpolation='none')

    plt.savefig(save_figure_path + "_reconstruct.png")
    plt.close()

def elbow(z):
    fig, ax = plt.subplots()
    distortions = []
    for i  in range(1,100):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=args.random_state)
        km.fit(z)
        distortions.append(km.inertia_)

    plt.plot(range(1,100),distortions,marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.savefig(save_figure_path + "_elbow.png")
    plt.close()

@hydra.main(config_path="config", config_name="vae")
def main(args):

    transform = tf.Compose(
        [
            tf.CenterCrop(224),
        ]
    )
    target_transform = transforms.ToIndex(args.labels)

    num_classes = len(args.labels)

    dataset = datasets.HDF5(args.dataset_path, transform=transform, target_transform=target_transform)
    train_set, test_set = dataset.split(train_size=args.train_size, random_state=args.random_state, stratify=dataset.targets)

    if False:
#    if True:
        # reconstruct augmention image
        augment = tf.Compose(
            [
                tf.RandomAffine(0, translate=(0.088, 0), fillcolor=None),
                tf.CenterCrop(224),
            ]
        )
        test_set.transform = augment

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(args.gpu.eval)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    is_output_gaussian = False
    model = models.VAE(args.in_channels, args.z_dim, is_output_gaussian=is_output_gaussian).to(device)
    figure_output_dir = os.path.dirname(args.trained_model_file)
    figure_model_epoch = os.path.basename(args.trained_model_file).split('.pt')[0]
    save_figure_path = figure_output_dir + '/' + figure_model_epoch
    try:
        model.load_state_dict_part(torch.load(args.trained_model_file))
    except:
        raise FileNotFoundError(f"Model file does not exist: {args.trained_model_file}")

    z, y = [], []
    model.eval()
    is_plot_reconstruct = True
    with torch.no_grad():
        for x, target in tqdm(test_loader):
            x = x.to(device, non_blocking=True)
            if is_output_gaussian:
                z_m, x_rec, _ = model.params(x)
            else:
                z_m, x_rec = model.params(x)
            z.append(z_m)
            y.append(target)
            # plot once
            if is_plot_reconstruct:
                plotReconsructImages(x, x_rec, save_figure_path)
                is_plot_reconstruct = False
    z = torch.cat(z).cpu().numpy()
    y = torch.cat(y).cpu().numpy().astype(int)


    # plot 2D
    if False:
        print(f"Plotting 2D latent features with true labels...")
        z_tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, n_iter=2000, random_state=args.random_state).fit(z).embedding_

        fig, ax = plt.subplots()
        #cmap = F.segmented_cmap("tab10", num_classes)
        cmap = F.segmented_cmap("tab20b", num_classes)
        for i in range(num_classes):
            idx = np.where(y == i)[0]
            if len(idx) > 0:
                c = cmap(i)
                ax.scatter(z_tsne[idx, 0], z_tsne[idx, 1], color=c, label=args.labels[i], edgecolors=F.darken(c))
        ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
        ax.set_aspect(1.0 / ax.get_data_ratio())
        ax.set_xlabel('t-SNE component 1')
        ax.set_ylabel('t-SNE component 2')
        plt.tight_layout()
        plt.savefig(save_figure_path + f"_z_tsne_perp{args.tsne_perplexity}.png", dpi=300)
        plt.close()
    else:
        print(f"Plotting 3D latent features with true labels...")
        z_tsne = TSNE(n_components=3, perplexity=args.tsne_perplexity, n_iter=2000, random_state=args.random_state).fit(z).embedding_
        fig = plt.figure(figsize=(10, 10)).gca(projection='3d')
        cmap = F.segmented_cmap("tab20b", num_classes)
        for i in range(num_classes):
            idx = np.where(y == i)[0]
            if len(idx) > 0:
                c = cmap(i)
                fig.scatter(z_tsne[idx, 0], z_tsne[idx, 1], z_tsne[idx, 2], color=c, label=args.labels[i], edgecolors=F.darken(c))
        fig.legend(bbox_to_anchor=(0.0, 1.1), loc='upper left', ncol=4, fontsize=11)
        fig.set_xlabel('\nt-SNE component 1', linespacing=3.0)
        fig.set_ylabel('\nt-SNE component 2', linespacing=3.0)
        fig.set_zlabel('\nt-SNE component 3', linespacing=3.0)
        fig.set_xlim(-25,25)
        fig.set_ylim(-25,25)
        fig.set_zlim(-25,25)
        # transparent background
        fig.w_xaxis.set_pane_color((0., 0., 0., 0.))
        fig.w_yaxis.set_pane_color((0., 0., 0., 0.))
        fig.w_zaxis.set_pane_color((0., 0., 0., 0.))
        plt.tight_layout()
        plt.savefig(save_figure_path + f"_z_tsne3_perp{args.tsne_perplexity}.png", dpi=300)

        fig.set_zlabel('\nt-SNE component 3', linespacing=3.0, rotation=90)
        fig.view_init(25, 20)
        plt.savefig(save_figure_path + f"_z_tsne3_perp{args.tsne_perplexity}_rotation.png", dpi=300)

        plt.close()

                
    
    print(f"Plotting silhouette coefficient of latent features ...")
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    silhouette_means = []
    silhouette_positions = []
    silhouette_colors = []

    silhouette_vals = silhouette_samples(z_tsne, y)

    fig, ax = plt.subplots(figsize=[18, 18])
    for i in np.unique(y)[::-1]:
        silhouette_vals_i = silhouette_vals[y == i]
        silhouette_vals_i.sort()
        silhouette_means.append(np.mean(silhouette_vals_i))
        y_ax_upper = y_ax_lower + len(silhouette_vals_i)
        c = cmap(i)
        plt.barh(
            range(y_ax_lower, y_ax_upper),
            silhouette_vals_i,
            height=1.0,
            edgecolor="none",
            color=c,
            alpha=0.8,
            zorder=1,
        )
        pos = (y_ax_lower + y_ax_upper) / 2
        silhouette_positions.append(pos)
        silhouette_colors.append(F.darken(c))

        y_ax_lower = y_ax_upper + 50  # 10 for the 0 samples

    ax.plot(silhouette_means, silhouette_positions, c="k", linestyle="dashed", linewidth=2.0, zorder=2)
    ax.axvline(np.mean(silhouette_vals), c="r", linestyle="dashed", linewidth=2.0, zorder=3)
    ax.legend(
        [
            Line2D([0], [0], c="r", linestyle="dashed", linewidth=2.0),
            Line2D([0], [0], color="k", linestyle="dashed", linewidth=2.0),
        ],
        ["average", "average for each label"],
        loc="upper left",
        fontsize=20
    )
    ax.set_ylim([0, y_ax_upper])
    ax.set_xlabel("Silhouette coefficient: average %f.1" % (np.mean(silhouette_vals)))
    plt.yticks(silhouette_positions, args.labels[::-1], rotation=0)
    plt.tight_layout()
    plt.savefig(save_figure_path + "_silhouette.png")
    plt.close()

if __name__ == "__main__":
    main()
