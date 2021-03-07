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


@hydra.main(config_path="config", config_name="iic")
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
    dataset = dataset.co(augment)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        sampler=torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=args.batch_size * args.num_train_step),
        pin_memory=True,
        drop_last=True,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(args.gpu.train)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    model = models.IIC(
        args.in_channels, num_classes=args.num_classes, num_classes_over=args.num_classes_over, z_dim=args.z_dim, num_heads=args.num_heads
    ).to(device)
    if args.pretrained:
        try:
            model.load_state_dict_part(torch.load(args.pretrained_model_file))
        except:
            raise FileNotFoundError(f"Model file does not exist: {args.pretrained_model_file}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger = logging.LossLogger(args)
    for epoch in range(args.num_epoch):
        print(f"training at epoch {epoch}...")
        model.train()
        num_samples = 0
        losses = np.zeros(3)
        for (x, x_), _ in tqdm(train_loader):
            x, x_ = x.to(device, non_blocking=True), x_.to(device, non_blocking=True)
            mi, mi_over = model(x, x_, lam=args.lam, z_detach=args.z_detach)
            loss = sum([mi, mi_over])
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses += np.array([loss.item(), mi.item(), mi_over.item()])
            num_samples += len(x)
        losses /= num_samples
        logger.update(total_train=losses[0], mi_train=losses[1], mi_over_train=losses[2])

        if epoch % args.save_itvl == 0:
            model_file = f"iic_e{epoch}.pt"
            torch.save(model.state_dict(), os.path.join(args.model_dir, model_file))
            print(f"Model parameters are saved to {model_file}.")

        if (epoch % args.eval_itvl == 0) and (args.wandb.is_output == False):
            for key, value in logger.items():
                logger.save(key, epoch, f"{key}_e{epoch}.png", xlabel="epoch", ylabel=key, xlim=(0, epoch))


if __name__ == "__main__":
    main()
