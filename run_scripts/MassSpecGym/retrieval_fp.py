""" Conduct fp-based retrieval for the mist model"""
import yaml
import pandas as pd
from pathlib import Path
import subprocess
import re
import yaml

dataset_name = "MassSpecGym"

model_inputs = [
    {
        "res_dir": f"results/{dataset_name}/mist3",
        "save_dir": f"results/{dataset_name}/mist3/mist_fp_retrieval/",
    },
]

num_workers = 16

labels_file = f"data/paired_spectra/{dataset_name}/labels.tsv"
subform = f"data/paired_spectra/{dataset_name}/subformulae/default_subformulae/"
spec = f"data/paired_spectra/{dataset_name}/spec_files/"

retrieval_db = f"data/paired_spectra/{dataset_name}/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db.h5"

#devices = ",".join(["1"])
#cuda_vis_str = f"CUDA_VISIBLE_DEVICES={devices}"

#fp_predict_base = f python3 src/mist/pred_fp.py --dataset-name {dataset_name} --num-workers {num_workers} --gpu --output-targs --subset-datasets test_only --labels-file {labels_file} --spec-folder {spec} --subform-folder {subform}"

fp_retrieval_base = f"""python src/mist/retrieval_fp.py \
    --dist-name cosine \
    --num-workers {num_workers} \
    --labels-file {labels_file} \
    --top-k 200 \
    --hdf-file {retrieval_db} \
"""

for test_dict in model_inputs:
    res_dir = Path(test_dict["res_dir"])
    save_dir = Path(test_dict["save_dir"])
    ckpts = list(res_dir.rglob("best.ckpt"))
    for ckpt in ckpts:
        # Predict fingerprints
        fold_name = ckpt.parent.name
        save_dir_temp = save_dir / f"{fold_name}"
        save_dir.mkdir(exist_ok=True, parents=True)
        #cmd = f"{fp_predict_base} --model-ckpt {str(ckpt)} --save-dir {save_dir_temp}"
        #print(cmd)
        #subprocess.call(cmd, shell=True)
        print(save_dir_temp)
        pred_file = save_dir.parent / "fp_preds" / f"fp_preds_{dataset_name}.p"

        # Run retrieval
        cmd = (
            f"{fp_retrieval_base} --fp-pred-file {pred_file} --save-dir {save_dir_temp}"
        )
        print(cmd)
        subprocess.call(cmd, shell=True)
