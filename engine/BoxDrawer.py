import cv2
from engine.DatabaseSearcher import DatabaseSearcher
class BoxDrawer:
    def __init__(self):
        self.database_searcher = DatabaseSearcher()
    def draw_boxes(self,boxes, frame):
        for result in boxes:
            x1, y1, width, height = result['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            cv2.rectangle(frame,(x1,y1),(x2,y2),color=(200,200,200),thickness=4)
            face = frame[y1:y2,x1:x2]
            name = self.database_searcher.search(face)
            # print(name)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (x1, y1-5)
            fontScale = 1
            fontColor = (200,200,200)
            lineType = 2

            cv2.putText(frame,name,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)