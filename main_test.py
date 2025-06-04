import h5py
with h5py.File("samples/s2con/s2con.ssl", "r") as f:
    f.visit(print)