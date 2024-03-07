""" lineplots.py

Create lineplots of retrieval

Example:

"""
import argparse
from pathlib import Path
import pickle
import pandas as pd

import numpy as np
from sklearn import metrics

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
    parser.add_argument("--model-names", nargs="+", default=[])
    return parser.parse_args()


def main():
    args = get_args()
    png = args.png
    model_names = args.model_names
    ext = "png" if png else "pdf"
    save_dir = args.save_dir
    retrieval_files = args.retrieval_files


    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    save_name = f"retrieval.{ext}"
    save_name = save_dir / save_name

    colors = np.array(mpl.cm.get_cmap("tab10").colors[:len(model_names)])

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

#    ax_figsize = (6, 7)
#    fig = plt.figure(figsize=(ax_figsize))
    fig = plt.figure()
    #plt.set_cmap('tab10')
#    plt.gca().set_aspect('equal')

    print(top_k_x.shape, top_k_y.shape)

    for x, y, model, c in zip(top_k_x, top_k_y, model_names, colors):
        auc = metrics.auc(x[:20], y[:20])
        plt.step(x[0:], y[0:], color=c, label=f"{model} - auc = {auc:.4f}")

    plt.xlim([0, 20])
    plt.ylim([0, 1])
    plt.xlabel("Top K")
    plt.ylabel("Accuracy")
        #ax.legend(loc='lower right', ncol=2, fontsize="15")

    #ax.tick_params(axis='x',labelsize=12)
    #ax.tick_params(axis='y',labelsize=12)
    #ax.xaxis.label.set_size(15)
    #ax.yaxis.label.set_size(15)

#    fig.legend(loc="lower center", ncol=len(model_names), fontsize="15")
    fig.legend()

#    set_size(*ax_figsize)
    fig.savefig(save_name, bbox_inches="tight", dpi=400, transparent=True)


if __name__ == "__main__":
    main()
