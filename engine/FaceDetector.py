from mtcnn.mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        self.model = MTCNN()
    def predict(self,input):
        return self.model.detect_faces(input)
    def get_faces(self,input):
        faces = self.model.detect_faces(input)
        result = []
        for face in faces:
            x1, y1, width, height = face['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            result.append(input[y1:y2,x1:x2])
        return result
