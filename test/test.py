import np2mcpl
import numpy as np
import pytest
import os
import glob
import hashlib
import pathlib as pl

l1 = np.linalg.norm( (4,5,6,) )
l2 = np.linalg.norm( (6,3,5,) )
prtcls_1 = np.array([[2112, 1., 2., 3.,   4./l1, 5./l1, 6./l1,    7., 9., 4., 0.,1.,0. ],
                     [22, 3., 2., 5.,   6./l2, 3./l2, 5./l2,    6., 3., 5., 1.,0.,0.]])
prtcls_2 = np.array([[11, 1., 2., 3.,   4./l1, 5./l1, 6./l1,    7., 9., 4.],
                     [-11, 3., 2., 5.,   6./l2, 3./l2, 5./l2,    6., 3., 5.]])

m5sums=[
  ['file1.mcpl','53c34894ec4312c69d79d6777c910ced',prtcls_1],
  ['file2.mcpl','70364cb5c1c75adc1e70079c4bd32f7b',prtcls_2],
]

def m5(fname):
  if (fname.endswith('.gz')):
    os.system(f'gunzip {fname}')
    fname=fname[:-3]
  hash_m5 = hashlib.md5()
  with open(fname, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_m5.update(chunk)
  return hash_m5.hexdigest()

def generate_file(f,pbank):
  fname=pl.Path(f).stem 
  np2mcpl.save(fname,pbank)
 
@pytest.mark.parametrize('f,exp_m5,particles',m5sums)
def test_single_file(f,exp_m5,particles):
  print(f)
  generate_file(f,particles) 
  os.system(f'gunzip {f}.gz')
  m5sum=m5(f)
  assert m5sum==exp_m5 
  os.unlink(f)
