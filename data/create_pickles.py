import os
import glob
import pickle
import cv2
from engine.FaceNet import FaceNet


def create_pickles():
    directory = os.path.join("..","data","dataset")
    # dirs = [x[0] for x in os.walk(directory)][1:]
    dirs = glob.glob(directory+"\\*")
    facenet = FaceNet()
    index = 0
    for dir in dirs:
        print(index,dir)
        index+=1
        images = []
        for j in range(6):
            name = str(j)+".jpg"
            image_path = os.path.join(dir,name)
            image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
            images.append(image)

        y = facenet.predict_batch(images)
        with open(os.path.join(dir,"data.pickle"),"wb") as file:
            pickle.dump(y,file)

create_pickles()