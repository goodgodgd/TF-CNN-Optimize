import os
import numpy as np
import scipy.io as sio


def convert_to_mat(npyfile):
  data = np.load(npyfile)
  matfile = npyfile.replace(".npy", ".mat")
  print(npyfile, "to", matfile)
  sio.savemat(matfile, {"data":data})

if __name__ == '__main__':

  data_dir = "/home/cideep/Work/tensorflow/output-data"

  for root, dirs, files in os.walk(data_dir, topdown=False):
    for name in files:
      if name.endswith(".npy"):
        convert_to_mat(os.path.join(root, name))
