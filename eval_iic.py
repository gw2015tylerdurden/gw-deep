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
import seaborn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, confusion_matrix

import src.data.datasets as datasets
import src.nn.models as models
import src.utils.transforms as transforms
import src.utils.functional as F
import src.utils.similarity_matrix as sm
import src.utils.logging as logging

plt.style.use("seaborn-poster")
plt.rcParams["lines.markersize"] = 6.0
plt.rc("legend", fontsize=10)

def PlotConfustionMatrix(cm, xlabelname, filename):
    fig, ax = plt.subplots()
    seaborn.heatmap(
        cm,
        ax=ax,
        linewidths=0.1,
        linecolor="gray",
        cmap="afmhot_r",
        cbar=True,
        cbar_kws={"aspect": 50, "pad": 0.01, "anchor": (0, 0.05)},
    )
    plt.yticks(rotation=45)
    plt.xlabel(xlabelname)
    plt.ylabel("true labels")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def CaluculateAccuracy(cm_y, labels_name):
    """
    Calculate accuracy from confusion matrix.

    Parameters

    cm_y : confustion matrix of a classifier. shape(true label num, pred label num)
    labels_name: target labels name

    labels_name are converted to abbreviations in this function, as bellow
    array(['1080Lines', '1400Ripples', 'Air.C', 'Blip', 'Chirp', 'Ext.L',
       'Helix', 'Koi.F', 'Lig.M', 'Low.F.B', 'Low.F.L', 'No.G',
       'None.o.t.A', 'Pai.D', 'Pow.L', 'Rep.B', 'Sca.L', 'Scratchy',
       'Tomte', 'Vio.M', 'Wan.L', 'Whistle'], dtype='<U11')

    Exsamples

    fix output indices for each column.
    given label = [0, 1, 2], output column is 5. ( ex. 2 has 3 classes )
    column_max_indices = np.argmax(cm_y, axis=0) [] -> [0, 2, 1, 2, 2]
    then, calculate accuracy of it
    """
    targets = np.array([F.acronym(target) for target in labels_name])

    accuracies = []
    column_max_indices = np.argmax(cm_y, axis=0)
    n_true, n_neg = 0, 0
    new_labels = []
    new_labels_counter = defaultdict(lambda: 0)
    for l, m in enumerate(column_max_indices):
        cm_column = cm_y[:, l]
        # set true value as excepted value of column_max_indices[m]
        n_true += cm_column[m]
        # set neg value as excepted value of excluded column_max_indices[m]
        n_neg += np.take(cm_column, [t for t in range(len(targets)) if t != m]).sum()
        if cm_column.sum() > 5:
            a = cm_column[m] / cm_column.sum()
            label = targets[m]
            new_labels.append(f"{l}:{label}-{new_labels_counter[label]}")
            print("new labels?")
            new_labels_counter[label] += 1
            accuracies.append(0)
    new_labels = np.array(new_labels)

    # divided column labels num
    accuracy = n_true / cm_y.sum()
    #gaccs.append(accuracies)
    return accuracy

def EnsambleConfucionMatrices(cms_list):
    """
    PRELIMINARY
    Ensamble confusion matrices(cms) of classifieres.
    Output labels of classifiers is randomly, so we couldn't ensamble them simply.
    This method sorts colmun of cms using cosine similarity,
    and then ensambles them.
    """

    # convert list to tensor
    cms = torch.FloatTensor(cms_list)

    num_heads = cms.shape[0]
    num_column = cms.shape[2]
    ensamble = torch.zeros(cms[0].shape)

    # for check index of confusion matrix
    check_idx = np.full([num_heads, num_column], -1)
    # calculate cosine similarity each column(dim=0)
    cosim = torch.nn.CosineSimilarity(dim=0)

    for target_column in range(num_column):
        idx = -1
        # set classifier 0
        target = cms[0, :, target_column]
        for head in range(num_heads):
            maxcosim = 0.
            classifier = cms[head, :, :]
            for column in range(num_column):
                search = classifier[:, column]
                tmp = cosim(target, search)
                # skip idx which is alreadly used in ensamble column
                if ((tmp > maxcosim) and (not (column in check_idx[head]))):
                    maxcosim = tmp
                    idx = column

            if idx == -1:
                # there is no max cosine similarity (0 vector situation)
                # skip summation and continue to next classifier
                continue

            # if maxcosim < threashold:
            #     # ignore low cosine similarity
            #     check_idx[head][idx] = -2
            #     continue

            ensamble[:, target_column] += classifier[:, idx] / num_heads
            check_idx[head][target_column] = idx

    print(f"Debug info: check_idx {check_idx}")
    return ensamble


@hydra.main(config_path="config", config_name="iic")
def main(args):

    transform = tf.Compose(
        [
            tf.CenterCrop(224),
        ]
    )

    target_transform = transforms.ToIndex(args.labels)

    dataset = datasets.HDF5(args.dataset_path, transform=transform, target_transform=target_transform)

    test_loader = torch.utils.data.DataLoader(
        dataset,
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

    model = models.IIC(
        args.in_channels, num_classes=args.num_classes, num_classes_over=args.num_classes_over, z_dim=args.z_dim, num_heads=args.num_heads
    ).to(device)
    model_file = os.path.join(args.model_dir, args.iic_trained_model_file)
    figure_output_dir = os.path.dirname(args.iic_trained_model_file)
    figure_model_epoch = os.path.basename(args.iic_trained_model_file).split('.pt')[0]
    try:
        model.load_state_dict_part(torch.load(args.iic_trained_model_file))
    except:
        raise FileNotFoundError(f"Model file does not exist: {args.iic_trained_model_file}")

    py, pw, z, y = [], [], [], []
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, target in tqdm(test_loader):
            x = x.to(device, non_blocking=True)
            py_m, pw_m, z_m = model.params(x) # [M, num class, Classifier num], [M, zdim]
            py.append(py_m)
            pw.append(pw_m)
            z.append(z_m)
            y.append(target) # [M] value is all 0 (or 1, or ...,C)
            num_samples += x.shape[0]
    # use in hypergraph
    classifiers_output_probability = py
    # get argmax of output column
    py = torch.cat(py).argmax(1).cpu().numpy().astype(int)  # shape: (N, num_heads), values: [0, num_classes-1]
    pw = torch.cat(pw).argmax(1).cpu().numpy().astype(int)  # shape: (N, num_heads), values: [0, num_classes_over-1]
    z = torch.cat(z).cpu().numpy() # shape(N, zdim)
    y = torch.cat(y).cpu().numpy().astype(int) # shape(N) ([ 0,  0,  0, ..., 21, 21, 21]) , not same class num
    accuracies = []
    hyper_graph = []
    print("")
    for i in range(args.num_heads):
        py_i, pw_i = py[:, i], pw[:, i]
        # cut y_true axis by arg.labels. [: len(args.labels), :]
        cm_y = confusion_matrix(y, py_i, labels=list(range(args.num_classes)), normalize="pred")[: len(args.labels), :]
        cm_w = confusion_matrix(y, pw_i, labels=list(range(args.num_classes_over)), normalize="pred")[: len(args.labels), :]

        # plot classes
        PlotConfustionMatrix(cm_y, "new labels", figure_output_dir + '/' + figure_model_epoch + '_' + f"cm_{i}.png")

        accuracy = CaluculateAccuracy(cm_y, args.labels)
        #print(f"accuracy= {accuracy:.3f} on classifier {i}")
        print(f"accuracy= {accuracy:.3f}")
        accuracies.append(accuracy)

        # plot overclasses
        PlotConfustionMatrix(cm_w, "new labels (overclustering)", figure_output_dir + '/' + figure_model_epoch + '_' + f"cm_over_{i}.png")
        #hyper_graph.append(cm_y)

    # cosine similarity
    simmat = sm.SimilarityMatrix(classifiers_output_probability, y, figure_output_dir + '/' + figure_model_epoch, args.num_classes, num_samples, args.labels, np.argmax(accuracies), args.random_state)
    #simmat = sm.SimilarityMatrix(hyper_graph, y, figure_output_dir + '/' + figure_model_epoch, args.num_classes, num_samples, args.labels, np.argmax(accuracies), args.random_state)
    print(f"sc_acc= {simmat.get_accuracy()}")
    simmat.plot_results()
    #simmat.plot_similarity_class(test_set)
    simmat.plot_predicted_similarity_class(dataset, is_plot_worse=True)


    # preliminary
    #ensamble_cm = EnsambleConfucionMatrices(confusion_matrices)
    #PlotConfustionMatrix(ensamble_cm, "new labels (ensamble)", figure_output_dir + '/' + figure_model_epoch + '_' + f"cm_ensamble.png")


if __name__ == "__main__":
    main()
