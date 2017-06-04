import pandas as pd

datapath = "/media/cideep/452B61C60A6D3269/tf-cnn-opt-2017/datasets/voc2012/VOC-2012-train/AnnotationsDF/AnnotationsDF.csv"
data = pd.read_csv(datapath, sep='\t')
label_names = data.category.unique()
label_names.sort()
print(label_names)

catg_num = []
for lname in label_names:
  labdata = data.loc[data['category'] == lname]
  catg_num.append(len(labdata))

print("total number of data \n", len(data))
print("number of data in each category \n", catg_num)
