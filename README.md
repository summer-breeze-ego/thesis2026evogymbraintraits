# Parent–child similarity of neural properties — thesis code

Code for a BSc thesis comparing two GRN-based controller encodings for
voxel soft robots in EvoGym (a binary-phase baseline vs. continuous,
GRN-derived per-voxel CPG parameters).

## Repository structure
- `evogym-GRN/` — the thesis code (GRN development, EA, EvoGym simulation, analysis notebooks)
- `evogym/` — the EvoGym library

## Dependencies
- **Python 3.9**
- **`numpy<2`** 
- **`opencv-python==4.9.0.80`** 
- `sqlalchemy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `gymnasium`, `lxml`, `cma`
- `statsmodels`
- EvoGym

## Data
To be able to reproduce the analysis, download the data from [[link](https://www.kaggle.com/datasets/augustincoman/thesis2026evogymbraintraits)] and extract it into evogym-GRN/tmp_out/thesis/

The two main analysis notebooks (analysis.ipynb, experiments/cmp500_brain_analysis.ipynb) read the database from there.
