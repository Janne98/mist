""" lineplots.py

Create lineplots of retrieval

Example:

"""
import argparse
from pathlib import Path
import pickle
import pandas as pd

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

from mist.utils.plot_utils import *

set_style()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-files",
        nargs="+",
        default=[
            "results/2022_08_31_retrieval_fig/csi_retrieval_cosine_0.p",
            "results/2022_08_31_retrieval_fig/csi_retrieval_Bayes.p",
            "results/2022_08_31_retrieval_fig/retrieval_contrast_pubchem_with_csi_retrieval_db_csi2022_cosine_0_ind_found.p",
            "results/2022_08_31_retrieval_fig/retrieval_contrast_pubchem_with_csi_retrieval_db_csi2022_cosine_0_merged_dist_0_7_ind_found.p",
        ],
    )
    parser.add_argument("--save-dir", default="results/retrieval_lineplots")
    parser.add_argument("--png", default=False, action="store_true")
    # parser.add_argument("--model-names", nargs="+")
    return parser.parse_args()


def main():
    args = get_args()
    png = args.png
    ext = "png" if png else "pdf"
    save_dir = args.save_dir
    retrieval_files = args.retrieval_files


    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    save_name = f"retrieval.{ext}"
    save_name = save_dir / save_name

    model_names = np.array(["MIST public", "MIST public", "MIST retrain", "MIST retrain", "MIST full", "MIST full", "CSI:FingerID", ])
    dist_names = np.array(["Cosine", "Contrastive", "Cosine", "Contrastive", "Cosine", "Contrastive", "Bayes"])
    colors = np.array(mpl.cm.get_cmap("tab20").colors[:len(model_names)])

    def shuffle_each_two_elements(arr):
        shuffled_arr = arr.copy()

        for i in range(0, len(arr) - 1, 2):
            temp = shuffled_arr[i].copy()
            shuffled_arr[i] = shuffled_arr[i + 1]
            shuffled_arr[i + 1] = temp

        return shuffled_arr

    colors = shuffle_each_two_elements(colors)

    # tuples of (model indices, k limit)
    models_to_plot = [([0, 1, 2, 3], 20), ([1, 3, 4, 5, 6], 20), ([1, 3, 4, 5, 6], 200)]

    assert len(retrieval_files) == len(dist_names)

    # Extract rankings from file
    ret_names, ret_inds = [], []
    for i in retrieval_files:
        with open(i, "rb") as fp:
            a = pickle.load(fp)
            ind_found, names = np.array(a["ind_found"]), np.array(a["names"])
            sort_order = np.argsort(names)
            names = names[sort_order]
            ind_found = ind_found[sort_order]

            ret_names.append(names)
            ret_inds.append(ind_found)

    # Calc common inds and subset
    common_inds = None
    for i, j in zip(ret_names, ret_inds):
        i = i[~np.isnan(j.astype(float))]
        temp_names = set(i)
        if common_inds is None:
            common_inds = temp_names
        else:
            common_inds = common_inds.intersection(temp_names)

    # Re-mask each based upon common inds
    new_names, new_inds = [], []
    for ret_name, ret_ind in zip(ret_names, ret_inds):
        mask = [i in common_inds for i in ret_name]
        new_names.append(ret_name[mask])
        new_inds.append(ret_ind[mask])
    ret_inds = new_inds
    ret_names = new_names

    # Create top k
    k_vals = np.arange(0, 1001)
    # max_k = np.max(k_vals) + 1
    top_k_x, top_k_y = [], []
    for ret_ind in ret_inds:
        new_x, new_y = [], []
        for k in k_vals:
            new_x.append(k)
            new_y.append(np.mean(ret_ind <= k))
        top_k_x.append(new_x), top_k_y.append(new_y)
    top_k_x, top_k_y = np.array(top_k_x), np.array(top_k_y)

    # ax_figsize = (16, 6)
    fig, axes = plt.subplots(1, len(models_to_plot), figsize=(9, 3.7))
    sns.set_context("paper")
    #plt.set_cmap('tab10')
    # ax = fig.gca()

    for (m_idx, lim), ax in zip(models_to_plot, axes):
        for x, y, model, dist, c in zip(top_k_x[m_idx], top_k_y[m_idx], model_names[m_idx], dist_names[m_idx], colors[m_idx]):
            ax.step(x[0:], y[0:], color=c, label=f"{model} - {dist}")
        ax.set_box_aspect(1)
        ax.set_xlim([0, lim])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Top K")
        # ax.set_ylabel("Accuracy")
        #ax.legend(loc='lower right', ncol=2, fontsize="15")

        # ax.tick_params(axis='x',labelsize=12)
        # ax.tick_params(axis='y',labelsize=12)
        # ax.xaxis.label.set_size(15)
        # ax.yaxis.label.set_size(15)
    axes[0].set_ylabel("Accuracy")

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # grab unique labels
    unique_labels = [f"{model} - {dist}" for model, dist in zip (model_names, dist_names)]
    legend_elements = [Patch(facecolor=c, edgecolor='w',label=l) for l, c in zip(unique_labels, colors)]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4) # , fontsize="15"
    
    # set_size(*ax_figsize)
    fig.savefig(save_name, bbox_inches="tight", dpi=400, transparent=True)


if __name__ == "__main__":
    main()
