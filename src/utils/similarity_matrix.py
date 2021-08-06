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
        Let argmax of a element of colmun be correct label
        """
        indices = np.argmax(self.cm, axis=0)
        n_true = 0
        for l, m in enumerate(indices):
            cmi = self.cm[:, l]
            n_true += cmi[m]
        sc_acc = n_true / self.cm.sum()
        return sc_acc

    # not yet...
    def _get_precision(self):
        indices = np.argmax(self.cmn, axis=1)
        precision = []
        for i in range(indices):
            coli = self.cmn[:, i]
            precision.append(coli / coli.sum())
        return precision

    def _get_recall(self):
        indices = np.argmax(self.cmn, axis=0)
        recall = []
        for i in range(indices):
            rowi = self.cmn[i, :]
            recall.append(rowi / rowi.sum())
        return recall

    def plot_results(self):
        self._plot_sc_confusion_matrix()
        self._plot_matirx(self.affinity_matrix, "affinity")
        self._plot_cosine_similarity_matirx(self.affinity_matrix)

    def plot_similarity_class(self, test_set, sample_top_similarity_num=5):
        samples_pred = F.sample_from_each_class(self.pred_sc_labels, sample_top_similarity_num)
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

    def plot_predicted_similarity_class(self, dataset, sample_top_similarity_num=5):
        # random sample each pred class, [[0, data idx], ..., [C, data idx]]
        #sample_each_pred_class = F.sample_from_each_class(self.pred_sc_labels, 1, np.random.randint(0, 256))

        cossim = torch.nn.CosineSimilarity(dim=0)
        for new_label_num in range(self.num_classes_expected):
            if new_label_num % 5 == 0:
                fig, ax = plt.subplots(nrows=sample_top_similarity_num, ncols=sample_top_similarity_num)

            new_label_dataset = np.where(self.pred_sc_labels == new_label_num)[0] # for 1 dimension[0]
            # select target rondomly
            random.seed()
            target_idx = random.choice(new_label_dataset)
            target_affinity_vector = torch.from_numpy(self.affinity_matrix[target_idx, :])
            # find similarity indices of tatgrt_affinity_vector
            affinity_ordered_newclass = []
            for idx, dataset_idx in enumerate(new_label_dataset):
                affinity_vector = torch.from_numpy(self.affinity_matrix[dataset_idx, :])
                cosine_similarity = cossim(affinity_vector, target_affinity_vector)

                affinity_ordered_newclass.append((dataset_idx, cosine_similarity))
                #affinity_dataset_ordered = sorted(affinity_dataset_ordered, key=lambda x: x[1])
                affinity_ordered_newclass.sort(key=lambda x: x[1], reverse=True)

            cols = new_label_num % 5
            sample = 0
            for idx in range(sample_top_similarity_num):
                # 0 is sample, 1 is most sim, 2 is 0.9 sim, ...
                sample = idx
                if idx > 1:
                    target_similarity_value = 1.0 - (0.1 * idx)
                    while(affinity_ordered_newclass[sample][1] > target_similarity_value):
                        sample += 1
                        if sample == len(affinity_ordered_newclass):
                            sample = idx
                            break

                x, _ = dataset[affinity_ordered_newclass[sample][0]]
                # plot 0.5sec glitch
                ax[cols, idx].imshow(x[0])
                ax[cols, idx].axis("off")
                ax[cols, idx].margins(0)

                if idx == 0:
                    # target of new label data
                    #ax.set_title(r"$x_{(%d)}$" % dataset_idx)
                    ax[cols, idx].set_title(r"$label {(%d)}$" % new_label_num, fontsize=17)
                else:
                    # similarity data of target class
                    # ax[cols, idx].set_title(r"data id[%d] %.2f" % (affinity_dataset_ordered[idx][0], affinity_dataset_ordered[idx][1]))
                    ax[cols, idx].set_title(r"sim %.2f" % (affinity_ordered_newclass[sample][1]), fontsize=17)

            if new_label_num % 5 == 4:
                plt.subplots_adjust(wspace=0.05, top=0.92, bottom=0.05, left=0.05, right=0.95)
                plt.savefig(self.filepath_part + f"_new_label_similar{new_label_num // 5}.png", transparent=True, dpi=300)
                plt.close()
            elif new_label_num == (self.num_classes_expected - 1):
                plt.subplots_adjust(wspace=0.05, top=0.92, bottom=0.05, left=0.05, right=0.95)
                plt.savefig(self.filepath_part + f"_new_label_similar{(new_label_num // 5) + 1}.png", transparent=True, dpi=300)
                plt.close()


    def __init__(self, pred_y, true_y, output_filepath, num_classes, num_samples, labels, top_accuracy_classifier_id=0, random_state=123, is_acronym=False):
        self.num_classes_expected = num_classes
        self.filepath_part = output_filepath
        if is_acronym:
            self.target_labels_name = np.array([F.acronym(target) for target in labels])
        else:
            self.target_labels_name = labels

        self.__calculate_confusion_matrix(pred_y, true_y, num_samples, random_state)

    def __calculate_confusion_matrix(self, pred_y, true_y, num_samples, random_state):
        # make hyper_graph from output of classifiers. shape [dataset N, heads*self.num_classes]
        concat_predicted_matrix_dataset = torch.cat(pred_y).view(num_samples, -1).cpu().numpy().astype(float)
        sc = SpectralClustering(n_clusters=self.num_classes_expected,
                                random_state=random_state,
                                assign_labels="discretize",
                                # assign_labels="kmeans",
                                affinity="rbf")
        spectral_cluster = sc.fit(concat_predicted_matrix_dataset)
        # affinity_matrix_.shape is [N, N]
        self.affinity_matrix = spectral_cluster.affinity_matrix_
        # plot eigenvalue, not neccesary
        #eigs, eigv = scipy.linalg.eigh(self.affinity_matrix)
        #self._plot_eigenvalues_similarity_matrix(eigs)

        # shape is [N, ]. array values are new class id.
        self.pred_sc_labels = spectral_cluster.labels_
        # confusion matrix is square [num_classes, num_classes]
        cm_squared = confusion_matrix(true_y, self.pred_sc_labels, labels=list(range(self.num_classes_expected)))
        # self.cm = cm_squared[np.nonzero(np.isin(np.arange(self.num_classes), true_y))[0], :]
        # changes row num of cm to num of gravity spy labels, [22, num_classes]
        self.cm = cm_squared[:len(self.target_labels_name), :]
        #self.cmn = normalize(self.cm, norm='l2', axis=0)
        self.cmn = normalize(self.cm, norm='l1', axis=0)
        # print(self._get_precision())
        # print(self._get_recall())

    def __old_calculate_confusion_matrix(self, pred_y, true_y, num_samples, random_state):
        # make hyper_graph from output of classifiers. shape [dataset N, heads*self.num_classes]
        concat_predicted_matrix_dataset = torch.cat(pred_y).view(num_samples, -1).cpu().numpy().astype(float)
        # make inferenced square matrix using cosine similarity. shape[N, N]
        inferenced_square_matrix = F.cosine_similarity(torch.from_numpy(concat_predicted_matrix_dataset)).numpy()
        # eigenvalue and eigenvector
        eigs, eigv = scipy.linalg.eigh(inferenced_square_matrix)
        sc = SpectralClustering(n_clusters=self.num_classes_expected,
                                random_state=random_state,
                                assign_labels="discretize",
                                # assign_labels="kmeans",
                                affinity="rbf")
        # choose num_classes of eigenvectors which are in ascending order, (last column vector is maximum)
        spectral_cluster = sc.fit(eigv[:, -self.num_classes_expected:])
        # shape is [N, ]. array values are new class id.
        self.pred_sc_labels = spectral_cluster.labels_
        # confusion matrix is square [num_classes, num_classes]
        cm_squared = confusion_matrix(true_y, self.pred_sc_labels, labels=list(range(self.num_classes_expected)))
        # self.cm = cm_squared[np.nonzero(np.isin(np.arange(self.num_classes), true_y))[0], :]
        # changes row num of cm to num of gravity spy labels, [22, num_classes]
        cm = cm_squared[:len(self.target_labels_name), :]
        self.cmn = normalize(cm, axis=0)

    def _plot_sc_confusion_matrix(self):
        fig, ax = plt.subplots()
        # classified datanum of each label
        datanum_each_label = [str(i) for i in self.cm.sum(axis=0)]

        # add row
        plot_data = np.insert(self.cmn, 0, np.zeros(len(datanum_each_label)), axis=0)
        plot_labels = np.insert(self.target_labels_name, 0, '')

        threshold_plot_rate = 0.1
        round_rate = np.round(self.cmn, decimals=2)
        annot_str = np.where(round_rate >= threshold_plot_rate, round_rate, '')
        annot_str = np.insert(annot_str, 0, datanum_each_label, axis=0)
        seaborn.heatmap(
            data=plot_data,
            ax=ax,
            #annot=self.cmn*self.cmn,
            annot=annot_str,
            annot_kws={"fontsize": 8},
            #fmt=".1f",
            fmt='',
            linewidths=0.1,
            cmap="Greens",
            cbar=True,
            cbar_kws={"aspect": 50, "pad": 0.01, "anchor": (0, 0.05)},
            yticklabels=plot_labels,
            xticklabels=np.arange(self.num_classes_expected),
        )
        ax.set_title(r"Number of data for each label")
        ax.set_xlabel(r"Unsupervised Labels")
        plt.tight_layout()
        plt.axhline(0, color='black', linewidth=2)
        plt.axhline(1, color='black', linewidth=2)
        plt.savefig(self.filepath_part + "_cm_sc.png", transparent=True, dpi=300)
        plt.close()

    def _plot_eigenvalues_similarity_matrix(self, eigen_values):
        fig, ax = plt.subplots()
        ax.plot(eigen_values.reshape(-1))
        ax.set_title("eigenvalues of dataset similarity matrix")
        ax.set_xlabel("eigenvalues in ascending order")
        ax.set_ylabel("eigen values")
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(self.filepath_part + "_eigen.png", dpi=300)
        plt.close()

    def _plot_matirx(self, matrix, filename):
        plt.pcolormesh(matrix)
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.savefig(self.filepath_part + "_" + filename + ".png", dpi=300)
        plt.close()


    def _plot_cosine_similarity_matirx(self, matrix):
        """
        distance_matrix shows that how close each data is.
        so this result helps to determine the true number of classes.
        distance_reordered is sorted indecies in descending.
        """
        distance_matrix, distance_reordered, _ = F.compute_serial_matrix(matrix)
        # sort predicted labels in distance
        reordered_pred_sc_labels = self.pred_sc_labels[distance_reordered]

        fig = plt.figure()
        axs = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0)
        axs[0].imshow(distance_matrix, aspect=1)
        # add newaxis to plot predicted labeles of glitches
        axs[1].imshow(reordered_pred_sc_labels[np.newaxis, :], aspect=70, cmap=F.segmented_cmap("tab20c", self.num_classes_expected))
        axs[0].axis("off")
        axs[1].axis("off")
        axs[0].set_title("cosine similarity matrix using Spectral-Clustering")
        plt.savefig(self.filepath_part + "_simmat_sc.png", transparent=True, dpi=300)
        plt.close()
