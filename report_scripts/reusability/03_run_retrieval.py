"""03_run_retrieval

run retrieval for the retrained mist models + make plots

"""
from collections import defaultdict
import pickle
import subprocess
from pathlib import Path
import shutil

mist_dir = "results/retrained_models/mist_fp_model_aug"
contrast_dir = "results/retrained_models/contrast_model_aug"
num_workers = 10
dataset_name = "casmi2022"
labels_file = f"data/paired_spectra/{dataset_name}/labels_true.tsv"
hdf_prefix = f"data/paired_spectra/{dataset_name}/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db"

fp_retrieval_base = f"python run_scripts/retrieval_fp.py --dist-name cosine --num-workers {num_workers} --labels-file {labels_file} --hdf-prefix {hdf_prefix}"
contrastive_retrieval_base = f"python3 run_scripts/retrieval_contrastive.py --dataset-name {dataset_name} --hdf-prefix {hdf_prefix} --num-workers 12 --dist-name cosine"

true_retrieval_ranking = f"data/paired_spectra/{dataset_name}/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db_ranked.p"
lam = 0.3
ensemble_script_base = f"python3 analysis/retrieval/avg_model_dists.py --lam {lam}"

pretrain_cos_inds = f"pretrained_models/mist_canopus_public/fp_model/retrieval_{dataset_name}/out_retrieval_single/retrieval_fp_intpubchem_with_morgan4096_retrieval_db_casmi2022_cosine_0_ind_found.p"
pretrain_contrast_inds = f"pretrained_models/mist_canopus_public/contrastive_model/retrieval_{dataset_name}/merged_retrieval_single/retrieval_avged_0_3_ind_found.p"

mist_full_cos_inds = f"pretrained_models/mist_full/fp_model/retrieval_{dataset_name}/out_retrieval_single/retrieval_fp_intpubchem_with_morgan4096_retrieval_db_casmi2022_cosine_0_ind_found.p"
mist_full_contrast_inds = f"pretrained_models/mist_full/contrastive_model/retrieval_{dataset_name}/merged_retrieval_single/retrieval_avged_0_3_ind_found.p"

csi_inds = "data/paired_spectra/casmi2022/prev_results/csi_ind_found.p"

# Whether we use single models or ensemble of models
merge_strat = "single"

# How to average FP models and contrastive models
ensemble_output_dir = Path(contrast_dir) / f"retrieval_{dataset_name}/merged_retrieval_{merge_strat}"
ensemble_output_dir.mkdir(exist_ok=True)

# Step 1 is run cosine retrieval for all splits fingerprint
sub_pred_dir = Path(mist_dir) / f"preds_{dataset_name}/out_preds_{merge_strat}"
fp_outdir = Path(mist_dir) / f"retrieval_{dataset_name}/out_retrieval_{merge_strat}"
fp_outdir.mkdir(exist_ok=True)
for pred_file in sub_pred_dir.glob("*.p"):  #("preds/*.p"):
    # Compute merged predictions
    cmd = f"{fp_retrieval_base} --fp-pred-file {pred_file} --save-dir {fp_outdir}"
    subprocess.call(cmd, shell=True)

# Step 2 is run contrastive retrieval for all splits
contrast_outdir = Path(contrast_dir) / f"retrieval_{dataset_name}/out_retrieval_all"
contrast_outdir.mkdir(exist_ok=True)
for ckpt in Path(contrast_dir).rglob("*.ckpt"):
    cmd = f"{contrastive_retrieval_base} --model-ckpt {ckpt} --save-dir {contrast_outdir} --labels-name labels_true.tsv"
    subprocess.call(cmd, shell=True)

# Move to separate file / folder if needed
split_to_dists = defaultdict(lambda: [])
for p_file in contrast_outdir.glob("*.p"):
    print(p_file)
    temp_out = pickle.load(open(p_file, "rb"))
    split_file = Path(temp_out['split_file']).stem
    split_to_dists[split_file].append(p_file)

ensembled_ret_files = []
for split, dist_files in split_to_dists.items():
    if merge_strat == "ensemble":
        dist_file = dist_files[0]
        save_name = ensemble_output_dir / f"retrieval_{split}.p"
        str_preds = " ".join([str(i) for i in dist_files])
        cmd = f"python analysis/retrieval/ensemble_model_dists.py --in-files {str_preds} --out-file {save_name}"
        #print(cmd)
        #subprocess.call(cmd, shell=True)
    elif merge_strat == "single":
        dist_file = dist_files[0]
        save_name = ensemble_output_dir / f"retrieval_{split}.p"
        shutil.copy2(dist_files[0], save_name)
    ensembled_ret_files.append(save_name)

# Cat all predictions here
merged_contrast_out = ensemble_output_dir / f"retrieval_cat_file.p"
in_file_str = " ".join([str(i) for i in ensembled_ret_files])
print(in_file_str)
cmd = f"python analysis/retrieval/cat_retrieval_preds.py --in-files {in_file_str} --out-file {merged_contrast_out}"
print(cmd)
subprocess.call(cmd, shell=True)

# Step 3 run ensembling between the two
contrast_file = merged_contrast_out
fp_file = [i for i in fp_outdir.glob("*.p") if "ind_found" not in str(i)][0]
lam_str = str(lam).replace(".", "_")
merged_file = ensemble_output_dir / f"retrieval_avged_{lam_str}.p"
cmd = f"{ensemble_script_base} --first-ranking {fp_file} --second-ranking {contrast_file} --save {merged_file}"
print(cmd)
subprocess.call(cmd, shell=True)

# Step 4 convert to ranked retrieval numbers
ranked_retrieval_extract_base = f"python3 analysis/retrieval/extract_rankings.py --true-ranking {true_retrieval_ranking} --labels {labels_file}"
out_ind_files = []
for name, j in zip(
        ["fp", "contrast", "merged"],
        [fp_file, contrast_file, merged_file]):
    save_name = j.parent / f"{j.stem}_ind_found.p"
    out_ind_files.append(save_name)
    cmd = f"{ranked_retrieval_extract_base} --ranking {j} --save-name {save_name}"
    subprocess.call(cmd, shell=True)

# Also get numbers for broad
for name, j in zip(
        ["fp", "contrast", "merged"],
        [*out_ind_files]):
    print(f"Retrieval for {name}")
    summary_cmd = f"python3 analysis/retrieval/ranking_summary.py --ranking-file {j}"
    subprocess.call(summary_cmd, shell=True)

# Step 5 Make plots
# save_dir = ensemble_output_dir / "plots"
mist_cos_inds = out_ind_files[-3]
mist_contrast_inds = out_ind_files[-1]
#plt_cmd_1 = f"python3 analysis/retrieval/lineplots.py --retrieval-files {csi_cos_inds} {csi_bayes_inds} {mist_cos_inds} {mist_contrast_inds} --save-dir {save_dir}"
# plt_cmd_1 = f"python3 analysis/retrieval/lineplots.py --retrieval-files {pretrain_cos_inds} {pretrain_contrast_inds} {mist_cos_inds} {mist_contrast_inds} {csi_inds} {mist_full_cos_inds} {mist_full_contrast_inds} --save-dir {save_dir} --png"
# print(plt_cmd_1)
# subprocess.call(plt_cmd_1, shell=True)

#plt_cmd_2 = f"python3 analysis/retrieval/retrieval_barplot.py --retrieval-files {csi_bayes_inds} {mist_contrast_inds} --save-name {save_dir / 'barplot.png'} --png"
# plt_cmd_2 = f"python3 analysis/retrieval/retrieval_barplot.py --retrieval-files {pretrain_contrast_inds} {mist_contrast_inds} --save-name {save_dir / 'barplot.png'} --png"
# subprocess.call(plt_cmd_2, shell=True)

plt_cmd_3 = f"python3 report_scripts/reusability/retrieval_lineplots.py \
    --retrieval-files \
        {pretrain_cos_inds} \
        {pretrain_contrast_inds} \
        {mist_cos_inds} \
        {mist_contrast_inds} \
        {mist_full_cos_inds} \
        {mist_full_contrast_inds} \
        {csi_inds} \
    --save-dir results/reusability/retrieval_lineplots \
    --png"
subprocess.call(plt_cmd_3, shell=True)
