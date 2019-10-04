# MNA-Faces

## Setup

It is highly recommended to use Anaconda (Python 3).
OpenCV is required.

For Ubuntu:
```bash
conda config --set allow_conda_downgrades true
conda install conda=4.6.14
conda install -c conda-forge opencv
```

## Execution

```bash
mode="pca" # or "kpca"
python main.py $mode
```

A file `eigenfaces_${mode}.txt` will be generated to save the state of the training.
That file will then be loaded once `main.py` is ran again.
To train again, simply delete the file for the specified mode.
