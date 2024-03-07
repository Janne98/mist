""" 04_create_lookup.py

create retrieval hdfs for casmi2022


"""
import mist.retrieval_lib.make_hdf5 as make_hdf5
import mist.utils as utils


PUBCHEM_FILE = "data/raw/pubchem/cid_smiles.txt"
PUBCHEM_FORMULA = "data/raw/pubchem/pubchem_formulae_inchikey.p"

DATASET_NAME = "casmi2022"
LABELS_FILE = "labels_true.tsv"

num_workers = 11


if __name__=="__main__":
    
    # Construct lookup library
    print("make retrieval hdf file")
    make_hdf5.make_retrieval_hdf5_file(
        dataset_name=DATASET_NAME, labels_name=LABELS_FILE,
        form_to_smi_file="data/raw/pubchem/pubchem_formulae_inchikey.p",
        database_name="intpubchem",
        fp_names=["morgan4096"],
        debug=False,
    )

    # Make a retrieval ranking file
    print("make retrieval ranking file")
    make_hdf5.make_ranking_file(
        dataset_name=DATASET_NAME,
        hdf_prefix=f"data/paired_spectra/{DATASET_NAME}/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db",
        labels_name=LABELS_FILE,
        num_workers=num_workers
    )

    # Subsample for contrastive learning
    print("subsample for contrastive learning")
    make_hdf5.subsample_with_weights(
        hdf_prefix=f"data/paired_spectra/{DATASET_NAME}/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db",
        labels_file=f"data/paired_spectra/{DATASET_NAME}/{LABELS_FILE}",
        fp_names=["morgan4096"],
        debug=False,
        num_workers=num_workers
    )
