import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import scipy
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import silhouette_samples, confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from . import functional as F
import random

class SimilarityMatrix():
    def get_accuracy(self):
        """
        calculate ensambled accuracy
        """
        indices = np.argmax(self.cm, axis=0)
        n_true = 0
        for l, m in enumerate(indices):
            cmi = self.cm[:, l]
            n_true += cmi[m]
        ensambled_acc = n_true / self.cm.sum()
        return ensambled_acc

    def plot_results(self):
        self._plot_ensambled_confusion_matrix()
        self._plot_eighen_similarity_matrix()
        self._plot_cosine_similarity_matirx()

    def plot_similarity_class(self, test_set, sample_top_similarity_num=5):
        samples_pred = F.sample_from_each_class(self.pred_ensamble, sample_top_similarity_num)
        for i, (label, indices) in enumerate(samples_pred.items()):
            if i % 5 == 0:
                fig, _ = plt.subplots()
                print(f"Plotting samples from each predicted classes {i // 5}...")
            for n, m in enumerate(indices):
                x, _ = test_set[m]
                ax = plt.subplot(5, sample_top_similarity_num, sample_top_similarity_num * (i % 5) + n + 1)
                ax.imshow(x[0])
                ax.axis("off")
                ax.margins(0)
                ax.set_title(r"$x_{(%d)} \in y_{(%d)}$" % (m, label))
            if i % 5 == 4:
                plt.subplots_adjust(wspace=0.05, top=0.92, bottom=0.05, left=0.05, right=0.95)
                fig.suptitle("Random samples from each predicted labels")
                plt.savefig(self.filepath_part + f"_samples_{i // 5}.png", transparent=True, dpi=300)
                plt.close()

        print(f"Plotting random samples with 5 most similar samples...")
        fig, _ = plt.subplots()
        sample_indices = random.sample(range(len(test_set)), 5)
        for i, j in enumerate(sample_indices):
            x, _ = test_set[j]
            ax = plt.subplot(len(sample_indices), sample_top_similarity_num + 2, (sample_top_similarity_num + 2) * i + 1)
            ax.imshow(x[0])
            ax.axis("off")
            ax.margins(0)
            ax.set_title(r"$x_{(%d)}$" % j)
            sim, sim_indices = torch.sort(torch.from_numpy(self.inferenced_square_matrix[j, :]), descending=True)
            sim, sim_indices = sim[1: sample_top_similarity_num + 1], sim_indices[1: sample_top_similarity_num + 1]
            for n, m in enumerate(sim_indices):
                ax = plt.subplot(len(sample_indices), sample_top_similarity_num + 2, (sample_top_similarity_num + 2) * i + 3 + n)
                x, _ = test_set[m]
                ax.imshow(x[0])
                ax.axis("off")
                ax.margins(0)
                ax.set_title(r"%.2f" % sim[n])
        plt.subplots_adjust(wspace=0.05, top=0.92, bottom=0.05, left=0.05, right=0.95)
        fig.suptitle("Random samples with corresponding similar glitches")
        plt.tight_layout()
        plt.savefig(self.filepath_part + f"_simrank.png", transparent=True, dpi=300)
        plt.close()


    def __init__(self, pred_y, true_y, output_filepath, num_classes, num_samples, labels, top_accuracy_classifier_id=0, random_state=123):
        self.num_classes = num_classes
        self.filepath_part = output_filepath
        self.targets = np.array([F.acronym(target) for target in labels])
        self.__calculate_similarity_matrix(pred_y, true_y, num_samples, random_state)

    def __calculate_similarity_matrix(self, pred_y, true_y, num_samples, random_state):
        # make hyper_graph from output of classifiers. shape [dataset N, heads*self.num_classes]
        concat_predicted_matrix_dataset = torch.cat(pred_y).view(num_samples, -1).cpu().numpy().astype(float)
        self.inferenced_square_matrix = F.cosine_similarity(torch.from_numpy(concat_predicted_matrix_dataset)).numpy()
        # make inferenced square matrix using cosine similarity. shape[N, N]
        self.eigs, self.eigv = scipy.linalg.eigh(self.inferenced_square_matrix)
        self.distance_matrix, self.reordered, _ = F.compute_serial_matrix(self.inferenced_square_matrix)
        sc = SpectralClustering(n_clusters=self.num_classes,
                                random_state=random_state,
                                assign_labels="discretize",
                                affinity="rbf")
        # generate labels which shape is [N, ]
        self.pred_ensamble = sc.fit(self.eigv[:, -self.num_classes:]).labels_
        cm = confusion_matrix(true_y, self.pred_ensamble, labels=list(range(self.num_classes)))
        self.cm_labels = np.unique(np.concatenate([true_y, self.pred_ensamble]))
        self.cm = cm[np.nonzero(np.isin(self.cm_labels, true_y))[0], :]
        self.cmn = normalize(self.cm, axis=0)


    def _plot_ensambled_confusion_matrix(self):
        fig, ax = plt.subplots()
        seaborn.heatmap(
            self.cmn,
            ax=ax,
            annot=self.cm,
            fmt="d",
            linewidths=0.1,
            cmap="Greens",
            cbar=False,
            yticklabels=self.targets,
            xticklabels=self.cm_labels,
            annot_kws={"fontsize": 8},
        )
        plt.yticks(rotation=45)
        ax.set_title(r"confusion matrix y with q(y) ensembled with SC")
        plt.tight_layout()
        plt.savefig(self.filepath_part + "_cm_sc.png", transparent=True, dpi=300)
        plt.close()

    def _plot_eighen_similarity_matrix(self):
        fig, ax = plt.subplots()
        ax.plot(self.eigs[::-1])
        ax.set_xlim(0, len(self.eigs) - 1)
        ax.set_title("eigh values of similarity matrix ")
        ax.set_xlabel("order")
        ax.set_ylabel("eigen values")
        ax.set_xlim((0, 100 - 1))
        ax.set_yscale("log")
        ax.set_ylim((1e-3, None))
        plt.tight_layout()
        plt.savefig(self.filepath_part + "_eigen.png", transparent=True, dpi=300)
        plt.close()

    def _plot_cosine_similarity_matirx(self):
        fig = plt.figure()
        axs = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0)
        axs[0].imshow(self.distance_matrix, aspect=1)
        axs[0].axis("off")
        axs[1].imshow(self.pred_ensamble[self.reordered][np.newaxis, :], aspect=100, cmap=F.segmented_cmap("tab20b", self.num_classes))
        axs[1].axis("off")
        axs[0].set_title("cosine similarity matrix with SC clusters")
        plt.savefig(self.filepath_part + "_simmat_sc.png", transparent=True, dpi=300)
        plt.close()
