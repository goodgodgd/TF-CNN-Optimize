import numpy as np
import pandas as pd

split_ratio = [0.7, 0.15, 0.15]
df_file_name = '/home/cideep/Work/tensorflow/datasets/Link-to-datasets/voc2012/' \
               + 'VOC-2012-train/AnnotationsDF/AnnotationsDF.csv'
print(df_file_name)
annots = pd.read_csv(df_file_name, sep='\t')
print("data length", len(annots))

length = len(annots)
to_train = int(length * split_ratio[0])
to_val = int(length * (split_ratio[0] + split_ratio[1]))
shuffled_indices = np.arange(length)
np.random.shuffle(shuffled_indices)
train_range = range(0,to_train)
val_range = range(to_train,to_val)
test_range = range(to_val,length)
print(val_range)

traind_inds = shuffled_indices[train_range]
print(traind_inds)
