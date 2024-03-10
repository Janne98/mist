import argparse
import subprocess
from collections import defaultdict
from pathlib import Path
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1, type=int)
args = parser.parse_args()
seed = args.seed

models = [
    ("full_model", f"results/ablation/ablation_retrieval{seed}/fp_full", f"results/ablation/ablation_retrieval{seed}/contrast_full"),
    ("no_magma", f"results/ablation/ablation_retrieval{seed}/fp_magma", f"results/ablation/ablation_retrieval{seed}/contrast_magma"),
    ("no_pairwise_differences", f"results/ablation/ablation_retrieval{seed}/fp_pairwise", f"results/ablation/ablation_retrieval{seed}/contrast_pairwise"),
    ("no_simulated_data", f"results/ablation/ablation_retrieval{seed}/fp_simulated", f"results/ablation/ablation_retrieval{seed}/contrast_simulated"),
    ("no_unfolding", f"results/ablation/ablation_retrieval{seed}/fp_unfolding", f"results/ablation/ablation_retrieval{seed}/contrast_unfolding"),
]

dataset = "casmi2022"
lam = 0.3

labels_file = "labels_true.tsv"
labels_path = f"data/paired_spectra/{dataset}/labels_true.tsv"
ranking_file = f"data/paired_spectra/{dataset}/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db_ranked.p"
hdf_prefix = f"data/paired_spectra/{dataset}/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db"

predict_base_cmd = f"python3 run_scripts/pred_fp.py --dataset-name {dataset} --labels-name {labels_file} --output-targs"
cosine_base_cmd = f"python3 run_scripts/retrieval_fp.py --dist-name cosine --labels-file {labels_path} --hdf-prefix {hdf_prefix}"
contrast_base_cmd = f"python3 run_scripts/retrieval_contrastive.py --dataset-name {dataset} --hdf-prefix {hdf_prefix} --dist-name cosine"

retrieval_results = []
for model_name, fp_dir, contrast_dir in models: 
    print(f"Running {model_name}")

    num_workers = 0 # MacOS: num_workers must be 0
    # Run fingerprint predictions
    fp_ckpt = list(Path(fp_dir).rglob("*.ckpt"))[0]
    fp_save_dir = Path(fp_dir) / "out_preds"
#    fp_save_dir.unlink()
    predict_cmd = f"{predict_base_cmd} --model-ckpt {fp_ckpt} --save-dir {fp_save_dir} --num-workers {num_workers}"
    subprocess.call(predict_cmd, shell=True)

    num_workers = 11
    # Run fingerprint retrieval
    cosine_save_dir = fp_save_dir.parent / "out_retrieval"
    cosine_save_dir.mkdir(exist_ok=True)
#    cosine_save_dir.unlink()

    pred_file = list(fp_save_dir.glob("*.p"))[0]
    cosine_cmd = f"{cosine_base_cmd} --fp-pred-file {pred_file} --save-dir {cosine_save_dir} --num-workers {num_workers}"
    subprocess.call(cosine_cmd, shell=True)

    # Run contrastive retrieval
    contrast_save_dir = Path(contrast_dir) / "out_retrieval"
    contrast_save_dir.mkdir(exist_ok=True)
#    contrast_save_dir.unlink()

    contrast_ckpt = list(Path(contrast_dir).rglob("*.ckpt"))[0]
    contrast_cmd = f"{contrast_base_cmd} --model-ckpt {contrast_ckpt} --save-dir {contrast_save_dir} --labels-name {labels_file} --num-workers {num_workers}"
    subprocess.call(contrast_cmd, shell=True)

    # Ensemble contrastive and fingerprint distances
    lam_str = str(lam).replace(".", "_")
    fp_ranking = list(cosine_save_dir.rglob("retrieval_*.p"))[-1]
    contrast_ranking = list(contrast_save_dir.rglob("retrieval_*.p"))[-1]
    merged_ranking = Path(contrast_dir) / "merged_retrieval" / f"merged_ranking_{lam_str}.p"

    avg_dists_cmd = f"python3 analysis/retrieval/avg_model_dists.py --lam {lam} --first-ranking {fp_ranking} --second-ranking {contrast_ranking} --save {merged_ranking}"
    subprocess.call(avg_dists_cmd, shell=True)

    # Extract rankings
    extract_base_cmd = f"python3 analysis/retrieval/extract_rankings.py --true-ranking {ranking_file} --labels {labels_path}"

    out_ind_files = []
    for name, j in zip (
        ["fp", "contrast", "merged"],
        [fp_ranking, contrast_ranking, merged_ranking]):
        save_name = j.parent / f"{j.stem}_ind_found.p"
        out_ind_files.append(save_name)
        extract_cmd = f"{extract_base_cmd} --ranking {j} --save {save_name}"
        subprocess.call(extract_cmd, shell=True)

    retrieval_results.append((model_name, out_ind_files[-1])) # only contrastive retrieval file

# Plot retrieval accuracy
model_names = " ".join([i[0] for i in retrieval_results])
retrieval_files = " ".join([str(i[1]) for i in retrieval_results])

plot_cmd = f"python3 report_scripts/ablation_retrieval/retrieval_lineplots.py --retrieval-files {retrieval_files} --model-names {model_names} --save-dir results/ablation/ablation_retrieval{seed} --png"
subprocess.call(plot_cmd, shell=True)
