import np2mcpl
import numpy as np
import pytest
import os
import glob
import hashlib


system_p = np.array([[2112, 1., 2., 3.,   4./l1, 5./l1, 6./l1,    7., 9., 4., 0.,1.,0. ],
                     [22, 3., 2., 5.,   6./l2, 3./l2, 5./l2,    6., 3., 5., 1.,0.,0.]])
system_up = np.array([[11, 1., 2., 3.,   4./l1, 5./l1, 6./l1,    7., 9., 4.],
                     [-11, 3., 2., 5.,   6./l2, 3./l2, 5./l2,    6., 3., 5.]])

b2checksums={
  'file1.mcpl':'27ba2c8c3a1b7dc2708f74cc55cb1536eb521516d513780f09daedf84b1d8dff9a8ba2a167e17e697f588483955ec0844ce871c7ae99a166d98855ce76644a91',
  'file2.mcpl':'eabedfe08bc3f6d1f8acc757a694db8259324862b668b13f240d41ba7f160f7912f6e8640e7c9cff300517a04b6099a7a13415f7e1099a0086463ac44e6730f1',
}

def generate_files():
  l1 = np.linalg.norm( (4,5,6,) )
  l2 = np.linalg.norm( (6,3,5,) )

  np2mcpl.save("file1",system_p)

  np2mcpl.save("file2",system_up)

def b2(fname):
  if (fname.endswith('.gz')):
    os.system(f'gunzip {fname}')
    fname=fname[:-3]
  hash_b2 = hashlib.blake2b()
  with open(fname, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_b2.update(chunk)
  return hash_b2.hexdigest()

def cleanup():
  files=glob.glob("file*.mcpl*")
  for f in files:
    os.unlink(f)
  
def test_small_files():
  generate_files()
  files=glob.glob("file*.mcpl.gz")
  for f in files:
    print(f)
    b2sum=b2(f)
    print(b2sum)
    assert b2sum==b2checksums[f[:-3]]
  cleanup() 
