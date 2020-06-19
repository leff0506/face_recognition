from keras.models import load_model
import numpy as np
class FaceNet:
    def __init__(self,path="model/facenet_keras.h5"):
        self.model = load_model(path)

    def predict(self,image):
        image = np.array(image)
        image = np.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2]))
        return self.model.predict(image)
    def predict_batch(self,images):
        images = np.array(images)
        return self.model.predict(images)