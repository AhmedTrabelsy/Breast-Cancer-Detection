import h5py

with h5py.File("breast_cancer_model.h5", "r") as f:
    print(list(f.keys()))
