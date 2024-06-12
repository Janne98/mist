# Process MassSpecGym smiles subsets
pubchem_file="data/unpaired_mols/MassSpecGym/MassSpecGym_candidates.txt"
pubchem_formula="data/unpaired_mols/MassSpecGym/MassSpecGym_formula_inchikey.p"
python3 src/mist/retrieval_lib/form_subsets.py \
    --input-smiles  $pubchem_file \
    --out-map $pubchem_formula
