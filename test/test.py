import np2mcpl
import numpy as np
import pytest
import os
import glob
import hashlib


l1 = np.linalg.norm( (4,5,6,) )
l2 = np.linalg.norm( (6,3,5,) )
system_p = np.array([[2112, 1., 2., 3.,   4./l1, 5./l1, 6./l1,    7., 9., 4., 0.,1.,0. ],
                     [22, 3., 2., 5.,   6./l2, 3./l2, 5./l2,    6., 3., 5., 1.,0.,0.]])
system_up = np.array([[11, 1., 2., 3.,   4./l1, 5./l1, 6./l1,    7., 9., 4.],
                     [-11, 3., 2., 5.,   6./l2, 3./l2, 5./l2,    6., 3., 5.]])

m5sums=[
  ['file1.mcpl','53c34894ec4312c69d79d6777c910ced'],
  ['file2.mcpl','70364cb5c1c75adc1e70079c4bd32f7b'],
]

def generate_files():

  np2mcpl.save("file1",system_p)

  np2mcpl.save("file2",system_up)

def m5(fname):
  if (fname.endswith('.gz')):
    os.system(f'gunzip {fname}')
    fname=fname[:-3]
  hash_m5 = hashlib.md5()
  with open(fname, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_m5.update(chunk)
  return hash_m5.hexdigest()

def cleanup():
  files=glob.glob("file*.mcpl*")
  for f in files:
    os.unlink(f)
  
def test_small_files():
  generate_files()
  files=glob.glob("file*.mcpl.gz")
  for f in files:
    os.system(f'gunzip {f}')
    m5sum=m5(f[:-3])
    assert m5sum==m5sums[f[:-3]]
  cleanup()

#test_small_files()
