import os
import hydra
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as tf
from collections import defaultdict
from tqdm import tqdm
from sklearn.manifold import TSNE

import src.data.datasets as datasets
import src.nn.models as models
import src.utils.transforms as transforms
import src.utils.functional as F
import src.utils.logging as logging

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

    if True:
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
            z_m, x_rec = model.params(x)
            z.append(z_m)
            y.append(target)
            # plot once
            if is_plot_reconstruct:
                plotReconsructImages(x, x_rec, save_figure_path)
                is_plot_reconstruct = False
    z = torch.cat(z).cpu().numpy()
    y = torch.cat(y).cpu().numpy().astype(int)


    print(f"Plotting 2D latent features with true labels...")
    z_tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, n_iter=5000, random_state=args.random_state).fit(z).embedding_
    fig, ax = plt.subplots()
    cmap = F.segmented_cmap("tab10", num_classes)
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


if __name__ == "__main__":
    main()
