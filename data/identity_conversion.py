from engine.FaceDetector import FaceDetector
from engine.FaceNet import FaceNet
import glob
import os
import cv2
import pickle
def convert():
    face_detector = FaceDetector()
    face_net = FaceNet("../model/facenet_keras.h5")
    identities = glob.glob("identities_images/*.jpg")
    for identity in identities:
        name = os.path.basename(identity)[:-4]
        image = cv2.imread(identity)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        face = face_detector.get_faces(image)[0]
        face = cv2.resize(face,(160,160))
        print(face.shape)
        X = face_net.predict(face)
        with open(os.path.join("identities",name+".pickle"),"wb") as file:
            pickle.dump(X,file)
convert()