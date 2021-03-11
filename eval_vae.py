import os
import hydra
import torch
import numpy as np
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

    model = models.VAE(args.in_channels, args.z_dim).to(device)
    figure_output_dir = os.path.dirname(args.trained_model_file)
    figure_model_epoch = os.path.basename(args.trained_model_file).split('.pt')[0]
    try:
        model.load_state_dict_part(torch.load(args.trained_model_file))
    except:
        raise FileNotFoundError(f"Model file does not exist: {args.trained_model_file}")

    z, y = [], []
    model.eval()
    with torch.no_grad():
        for x, target in tqdm(test_loader):
            x = x.to(device, non_blocking=True)
            z_m, _ = model.params(x)
            z.append(z_m)
            y.append(target)
    z = torch.cat(z).cpu().numpy()
    y = torch.cat(y).cpu().numpy().astype(int)

    print(f"Plotting 2D latent features with true labels...")
    z_tsne = TSNE(n_components=2, random_state=args.random_state).fit(z).embedding_
    fig, ax = plt.subplots()
    cmap = F.segmented_cmap("tab10", num_classes)
    for i in range(num_classes):
        idx = np.where(y == i)[0]
        if len(idx) > 0:
            c = cmap(i)
            ax.scatter(z_tsne[idx, 0], z_tsne[idx, 1], color=c, label=args.labels[i], edgecolors=F.darken(c))
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
    ax.set_aspect(1.0 / ax.get_data_ratio())
    plt.tight_layout()
    plt.savefig(figure_output_dir + '/' + figure_model_epoch + "_z_tsne.png")
    plt.close()


if __name__ == "__main__":
    main()
