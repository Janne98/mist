"""FP Predictions batched for repo.

Make FP predictions with the MIST pretrained model and MIST retrained model and process these
into figures.

"""

import subprocess
from collections import defaultdict
from pathlib import Path
import shutil

dirs = [
    "pretrained_models/mist_canopus_public/fp_model",
    "results/retrained_models/mist_fp_model_aug",
    # "results/retrained_models/ffn_fp_model",
    "pretrained_models/mist_full/fp_model",
]

labels_file = "labels_true.tsv"

num_workers = 0  # on macOS: not working with num_workers > 0
dataset_name = "casmi2022"
base_script = f"python3 run_scripts/pred_fp.py --dataset-name {dataset_name} --num-workers {num_workers} --labels-name {labels_file} --output-targs"

# Merge strat determines if we want to make single model or ensemble model
# predictions
# merge_strat = "ensemble"
merge_strat = "single"

outdirs = []
for dir_ in dirs:
    fold_to_fp = defaultdict(lambda: [])
    for ckpt in Path(dir_).rglob("best.ckpt"):
        fold_name = ckpt.parent.name
        savedir = ckpt.parent / f"preds_{dataset_name}"
        cmd = f"{base_script} --model-ckpt {str(ckpt)} --save-dir {savedir}"

        subprocess.call(cmd, shell=True)
        output_fp = list(savedir.glob("*.p"))[0]
        fold_to_fp[fold_name].append(output_fp)

    #
    outdir = Path(dir_) / f"preds_{dataset_name}"
    outdir.mkdir(exist_ok=True)
    if merge_strat == "ensemble":
        for k, v in fold_to_fp.items():
            str_preds = " ".join([str(i) for i in v])

            # Run merge predictions
            save_name = outdir / f"{k}_preds.p"
            cmd = f"python analysis/fp_preds/average_model_fp_preds.py --fp-files {str_preds} --save-name {save_name}"

            # average predictions
            subprocess.call(cmd, shell=True)

    elif merge_strat == "single":
        for k, v in fold_to_fp.items():
            ## Run merge predictions
            save_name = outdir / f"{k}_preds.p"
            shutil.copy2(v[0], save_name)
            # average predictions
    else:
        raise NotImplementedError()
    outdirs.append(outdir)

# for dir_ in outdirs:
#     dir_ = Path(dir_)
#     pred_files = list(dir_.glob("*.p"))
#     print(pred_files)
#     pred_files_str = " ".join([str(i) for i in pred_files if "merged" not in str(i)])
#     out_file = dir_ / f"merged_fp_preds.p"
#     out_file.parent.mkdir(exist_ok=True)
#     merge_str = f"python analysis/fp_preds/cat_fp_preds.py --in-files {pred_files_str} --out {out_file}"

#     print(merge_str)
#     subprocess.call(merge_str, shell=True)


# Conduct plotting
# pretrain_fp_file = f"{dirs[0]}/preds_{dataset_name}/out_preds_{merge_strat}/merged_fp_preds.p"
# retrain_fp_file = f"{dirs[1]}/preds_{dataset_name}/out_preds_{merge_strat}/merged_fp_preds.p"
# ffn_fp_file = f"{dirs[2]}/out_preds_{merge_strat}/preds/merged_fp_preds.p"
# res_folder = Path(retrain_fp_file).parent / "plots"
# res_folder.mkdir(exist_ok=True)

# Scatter plots
# cmds = [
#         f"python3 analysis/fp_preds/fp_scatter.py --fp-pred-file {retrain_fp_file} --csi-baseline {pretrain_fp_file} --metric Cosine --pool-method spectra --model-name MIST_retrain --png",
#         f"python3 analysis/fp_preds/fp_scatter.py --fp-pred-file {retrain_fp_file} --csi-baseline {pretrain_fp_file} --metric LL --pool-method spectra --model-name MIST_retrain --png",
#         f"python3 analysis/fp_preds/fp_scatter.py --fp-pred-file {retrain_fp_file} --csi-baseline {pretrain_fp_file} --metric LL --pool-method bit --model-name MIST_retrain --png",

#         f"python3 report_scripts/scatter_all.py --fp-pred-file {retrain_fp_file} --baseline {pretrain_fp_file} --out-dir results/reusability/fp_scatter",
#         f"python3 report_scripts/reusability/fp_violin_all.py --out-dir results/reusability/fp_violin --png",
# ]

# for cmd in cmds:
#     subprocess.call(cmd, shell=True)
