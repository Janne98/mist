""" Build retrieval hdf files for evaluation and contrastive learning"""

import mist.retrieval_lib.make_hdf5 as make_hdf5

dataset = "MassSpecGym"

if __name__ == "__main__":
    labels_file = f"data/paired_spectra/{dataset}/labels.tsv"
    make_hdf5.make_retrieval_hdf5_file(
        labels_file=labels_file,
        form_to_smi_file=f"data/unpaired_mols/{dataset}/{dataset}_formula_inchikey.p",
        output_dir=f"data/paired_spectra/{dataset}/retrieval_hdf/",
        database_name="MassSpecGym_candidates",
        fp_names=("morgan4096",),
        debug=False,
    )

    full_db = f"data/paired_spectra/{dataset}/retrieval_hdf/MassSpecGym_candidates_with_morgan4096_retrieval_db.h5"
    make_hdf5.export_contrast_h5(
        hdf_file=full_db,
        labels_file=labels_file,
        fp_names=("morgan4096",),
        subset_size=128,
        num_workers=64,
    )
