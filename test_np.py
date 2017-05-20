import numpy as np
import os
import shutil

indices = np.arange(10)
np.random.shuffle(indices)
print(indices)

a = 'asdf'
b = 'qwer'
c = [a, b]
print(c)

d = 'asdf%d asdf'
e = d
print(e)
e = d%1
print(e)

dstpath='/home/cideep/Work/tensorflow/output-data/cifar10_inception_resnet_v2'
if os.path.exists(dstpath):
  shutil.rmtree(dstpath, ignore_errors=True)
os.makedirs(dstpath)
filename = os.path.join(dstpath, 'probs.npy')

rnd = np.random.rand(3,3)
print(rnd)
np.save(filename, rnd)
loaded = np.load(filename)
print(loaded)

mat1 = np.random.rand(3,3)
vec1 = np.random.rand(3)
print('mat1', mat1)
print('vec1', vec1)
concat1 = np.column_stack((vec1, mat1))
concat2 = np.concatenate((np.expand_dims(vec1,1), mat1),axis=1)
concat3 = np.vstack((vec1, mat1))
print('concat1', concat1)
print('concat2', concat2)
print('concat3', concat3)

print(np.ones(10))