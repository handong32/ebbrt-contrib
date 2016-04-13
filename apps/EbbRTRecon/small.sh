#!/bin/sh
set -x
./hosted/build/Release/reconstruction -o 3TStackReconstruction.nii -i data/masked_stack-1.nii data/masked_stack-2.nii data/masked_stack-3.nii  data/masked_stack-4.nii --disableBiasCorrection --useAutoTemplate --useSINCPSF --resolution 2.0 --debug 0 --numThreads $1 --useCPU --iterations $2

