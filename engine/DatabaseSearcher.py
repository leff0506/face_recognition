import cv2
from engine.FaceDetector import FaceDetector
from engine.FaceNet import FaceNet
import glob
import pickle
import numpy as np
import os
class DatabaseSearcher:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_net = FaceNet()
        self.X = []
        self.Y = []
        self.fill_database()
    def fill_database(self):
        pickles = glob.glob("data/identities/*.pickle")
        for picklee in pickles:
            name = os.path.basename(picklee)
            with open(picklee,"rb") as file:
                x = pickle.load(file)
                self.X.append(x)
                self.Y.append(name)
    def search(self,face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (160, 160))
        # print(face.shape)
        X = self.face_net.predict(face)
        min_dist = 100
        answer = "error"
        for i in range(len(self.X)):
            d = np.linalg.norm(self.X[i]-X)
            # print('distance is {}' .format(d))
            if d < min_dist:
                answer = self.Y[i][:-7]
                min_dist = d
        if min_dist<= 1:
            return answer
        return "error"