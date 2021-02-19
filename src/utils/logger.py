import matplotlib.pyplot as plt


def setup(ax, **kwargs):
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


def loss(values, epoch, out, **kwargs):
    if epoch > 1:
        fig, ax = plt.subplots()
        epochs = np.linspace(0, epoch, len(values))
        ax.plot(epochs, values)
        setup(ax, **kwargs)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
