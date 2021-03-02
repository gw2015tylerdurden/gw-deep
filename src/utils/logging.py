import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import src.utils.functional as F

def setup(ax, **kwargs):
    plt.style.use("seaborn-poster")
    plt.rc("legend", fontsize=10)
    for k, v in kwargs.items():
        if k == "xlabel":
            ax.set_xlabel(v)
        elif k == "ylabel":
            ax.set_ylabel(v)
        elif k == "xlim":
            ax.set_xlim(v)
        elif k == "ylim":
            ax.set_ylim(v)
        elif k == "title":
            ax.set_title(v)


class LossLogger:
    def __init__(self, args):
        self.stats = defaultdict(list)
        if args.wandb.is_output:
            import wandb
            self.is_output_wandb = True
            wandb.init(project=args.wandb.project, group=args.wandb.group)
            wandb.run.name = args.wandb.name + "_" + wandb.run.name
            wandb.config.update(F.flatten(args))

    def update(self, **kwargs):
        if self.is_output_wandb:
            import wandb
            wandb.log(dict(kwargs))
        else:
            for key, value in kwargs.items():
                self.stats[key].append(value)

    def items(self):
        return self.stats.items()

    def save(self, key, epoch, out, **kwargs):
        yy = self.stats[key]
        if len(yy) > 1:
            fig, ax = plt.subplots()
            xx = np.linspace(0, epoch, len(yy))
            ax.plot(xx, yy)
            setup(ax, **kwargs)
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
