# """ Run magma on all training data"""
magma_file=src/mist/magma/run_magma.py
dataset=MassSpecGym

echo "Magma on subset"
python3 $magma_file \
--spectra-dir data/paired_spectra/$dataset/spec_files  \
--output-dir data/paired_spectra/$dataset/magma_outputs  \
--spec-labels data/paired_spectra/$dataset/labels.tsv  \
--max-peaks 50 \
--num-workers 64
#--debug
