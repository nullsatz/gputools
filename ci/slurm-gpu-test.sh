#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -t 00:40:00
#SBATCH -o gputools-gpu-test-%j.log
# Runs the test suite on a real GPU, which GitHub's runners do not have.
#
#   sbatch -p <gpu partition> ci/slurm-gpu-test.sh <conda env> [<conda env> ...]
#
# from the top of a gputools checkout. Each conda environment needs R,
# r-testthat, a compiler and the CUDA packages listed in INSTALL; the
# package is built against each environment's toolkit in turn.
set -u
src=$(pwd)
work=${TMPDIR:-/tmp}/gputools-gpu-test-$$
command -v conda >/dev/null || { echo "conda is not on the PATH"; exit 1; }
eval "$(conda shell.bash hook)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

status=0
for env in "$@"; do
  conda activate "$env" || { status=1; continue; }
  export CUDA_HOME=$CONDA_PREFIX
  echo "######## $env"
  rm -rf "$work" && mkdir -p "$work/lib"
  cp -r "$src" "$work/src"
  if ! R CMD INSTALL --library="$work/lib" "$work/src" > "$work/install.log" 2>&1; then
    tail -30 "$work/install.log"; status=1; continue
  fi
  # one R process per file, so a crash in one does not hide the others
  for f in "$work"/src/tests/testthat/test-*.R; do
    echo "-- $(basename "$f")"
    (cd "$work/src/tests/testthat" &&
      R_LIBS="$work/lib" Rscript -e "testthat::test_file('$f',
        package = 'gputools', load_package = 'installed',
        reporter = 'summary', stop_on_failure = TRUE)") || status=1
  done
  conda deactivate
done
rm -rf "$work"
exit $status
