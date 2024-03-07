SIRIUS=sirius.app/Contents/MacOS/sirius

CORES=11
INPUT_DIR=data/paired_spectra/casmi2022/spec_files/
OUTPUT_DIR=data/paired_spectra/casmi2022/sirius_outputs/

$SIRIUS --cores $CORES --output $OUTPUT_DIR --input $INPUT_DIR --naming-convention %filename --ignore-formula formula --ppm-max-ms2 10