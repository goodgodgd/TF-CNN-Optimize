import numpy as np

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