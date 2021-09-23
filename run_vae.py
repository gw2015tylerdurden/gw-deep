import os
import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as tf
from collections import defaultdict
from tqdm import tqdm

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

    # 24 / 272 ≒ 0.088
    augment = tf.Compose(
        [
            tf.RandomAffine(0, translate=(0.088, 0), fillcolor=None),
            tf.CenterCrop(224),
        ]
    )
    target_transform = transforms.ToIndex(args.labels)

    dataset = datasets.HDF5(args.dataset_path, transform=transform, target_transform=target_transform)
    # stratified split, split at the same ratio from each class
    train_set, test_set = dataset.split(train_size=args.train_size, random_state=args.random_state, stratify=dataset.targets)
    train_set.transform = augment

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=args.batch_size * args.num_train_step),
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(args.gpu.train)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    model = models.VAE(args.in_channels, args.z_dim, is_output_gaussian=False).to(device)
    #model = models.VAE(args.in_channels, args.z_dim, is_output_gaussian=True).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    # cwd is hydra.run.dir in yaml config
    args.wandb.cwd = model_dir = os.getcwd() + '/' + args.model_dir
    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)
    logger = logging.LossLogger(args)

    for epoch in range(args.num_epoch + 1):
        print(f"training at epoch {epoch}...")
        model.train()
        num_samples = 0
        losses = np.zeros(3)
        for x, _ in tqdm(train_loader):
            x = x.to(device, non_blocking=True)
            logp_xz, kl_gauss = model(x)
            loss = sum([logp_xz, kl_gauss])
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses = np.array([loss.item(), logp_xz.item(), kl_gauss.item()])
            num_samples += len(x)
        #losses /= num_samples
        logger.update(total_train=losses[0],
                      logp_xz_train=losses[1],
                      kl_train=losses[2])

        if epoch % args.save_itvl == 0:
            model_file = f"vae_e{epoch}.pt"
            torch.save(model.state_dict(), os.path.join(model_dir, model_file))
            print(f"Model parameters are saved to {model_file}.")

        if epoch % args.eval_itvl == 0:
            print(f"evaluating at epoch {epoch}...")
            model.eval()
            with torch.no_grad():
                losses = np.zeros(3)
                for x, target in tqdm(test_loader):
                    x = x.to(device, non_blocking=True)
                    logp_xz, kl_gauss = model(x)
                    loss = sum([logp_xz, kl_gauss])
                    losses = np.array([loss.item(), logp_xz.item(), kl_gauss.item()])
                #losses /= num_samples
                logger.update(total_eval=losses[0], logp_xz_eval=losses[1], kl_eval=losses[2])

                if args.wandb.is_output == False:
                    for key, value in logger.items():
                        logger.save(key, epoch, f"{key}_e{epoch}.png", xlabel="epoch", ylabel=key, xlim=(0, epoch))


if __name__ == "__main__":
    main()
