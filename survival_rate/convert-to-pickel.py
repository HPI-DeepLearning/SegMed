from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
import pickle
from glob import glob
import numpy as np
import pandas as pd
import pdb

def dir_to_dataset(glob_files, loc_train_labels=""):
    print("Gonna process:\n\t %s"%glob_files)
    dataset = []
    pdb.set_trace()
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
        print file_name
        print 'Are we in the loop ?'
        image = Image.open(file_name)
        img = Image.open(file_name).convert('LA') #tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
        if file_count % 10== 0:
            print("\t %s files processed"%file_count)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    if len(loc_train_labels) > 0:
        df = pd.read_csv(loc_train_labels)
        return np.array(dataset), np.array(df["Class"])
    else:
        return np.array(dataset)




Dataa, y = dir_to_dataset("image\\*.png","train.csv")
# Data and labels are read 

train_set_x = Dataa[:600]
val_set_x = Dataa[601:800]
test_set_x = Dataa[801:1000]
train_set_y = y[:600]
val_set_y = y[601:800]
test_set_y = y[801:1000]
# Divided dataset into 3 parts. I had 6281 images.

train_set = train_set_x, train_set_y
print 'Type of train_set_x',type(train_set_x)
print train_set_x
val_set = val_set_x, val_set_y
test_set = test_set_x, val_set_y

dataset = [train_set, val_set, test_set]

f = gzip.open('brain.pkl.gz','wb')
pickle.dump(dataset, f, protocol=2)
f.close()
