import cv2
import cv2 as cv
import face_recognition as fr
import numpy as np
import os

path = 'dataset/img'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)

for cls in mylist:
    curImg = cv.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print(classNames)


def findEncodings(images):

    encodeList = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Декодирование закончено')

cap = cv.VideoCapture('dataset/video/1.mp4')

while True:
    success, img = cap.read()
    imgS = cv.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    faceCurFrame = fr.face_locations(imgS)
    encodeCurFrame = fr.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = fr.compare_faces(encodeListKnown, encodeFace)
        faceDis = fr.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex]
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1, = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
            cv.putText(img, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv.imshow('Video', img)
        cv.waitKey(1)


