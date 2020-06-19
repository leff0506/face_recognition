import cv2
import glob
import os
from skimage import io
from engine.FaceDetector import FaceDetector
import random
import time
trusted = 0

face_detector = FaceDetector()

def download_all():
    celebrities = glob.glob("initial_dataset/*.txt")
    result = 0
    random.shuffle(celebrities)
    index =0
    for celebrity in celebrities[:100]:
        name = os.path.basename(celebrity)[:-4]

        if not os.path.exists(os.path.join("dataset",name)):
            os.makedirs(os.path.join("dataset",name))
        temp = download_celebrity(celebrity,os.path.join("dataset",name))
        print(index,name,temp)
        index +=1
        result+=temp
    return result
def download_celebrity(celebrity_file,result_dir):
    with open(celebrity_file,"r") as f:
        lines = f.readlines()
        random.seed = int(time.time())
        random.shuffle(lines)
    temp =0
    photos = []
    for line in lines:
        if temp == 6:
            break
        domen = line.split()[1]

        try:
            img = io.imread(domen)
            faces = face_detector.get_faces(img)
            if len(faces) == 1:
                temp +=1
                faces[0] = cv2.resize(faces[0],(160,160))
                photos.append(faces[0])
        except:
            pass
    if temp >=6:
        save_photos(photos, result_dir)
        return 1
    return 0

def save_photos(photos, result_dir):
    index = 0
    for photo in photos:
        cv2.imwrite(os.path.join(result_dir,str(index)+".jpg"),cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
        index +=1

print(download_all())
