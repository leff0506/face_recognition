import os
import glob
import pickle
import random
import numpy as np
def create_dataset():
    directory = os.path.join("..","data","dataset")
    # dirs = [x[0] for x in os.walk(directory)][1:]
    dirs = glob.glob(directory+"\\*")
    X = []
    Y = []
    index = 0
    for dir in dirs:
        print(index,dir)

        with open(os.path.join(dir,"data.pickle"),"rb") as file:
            y = pickle.load(file)
        a = list(range(1,6))
        for i in range(5):
            y1 = y[a[0]]
            y2 = y[a[1]]

            y_l1 = np.abs(y1-y2)
            X.append(y_l1)
            Y.append(1)
            random.shuffle(a)
        a = list(range(0,index))+list(range(index+1,len(dirs)))
        random.shuffle(a)

        for i in range(5):
            index_another = a[i]
            with open(os.path.join(dirs[index_another], "data.pickle"), "rb") as file:
                y_another = pickle.load(file)

            y1 = y[random.randint(0,5)]
            y2 = y_another[random.randint(0,5)]

            y_l1 = np.abs(y1-y2)
            X.append(y_l1)
            Y.append(0)
        index += 1
    X = np.array(X)
    Y = np.array(Y)
    with open(os.path.join("..","data","dataset.pickle"), "wb") as file:
        pickle.dump((X,Y),file)

create_dataset()