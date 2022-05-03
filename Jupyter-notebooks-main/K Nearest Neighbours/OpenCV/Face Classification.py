import numpy as np
import cv2
import os


# KNN

def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]

        d = distance(test, ix)
        dist.append([d, iy])

    dk = sorted(dist, key=lambda x: x[0])[:k]

    labels = np.array(dk)[:, -1]

    output = np.unique(labels, return_counts=True)

    index = np.argmax(output[1])
    return output[0][index]


skip = 0
dataset_path = './data/'

face_data = []
labels = []

class_id = 0 # Labels for the given file
names = {} # Map ID and file Name

# Data Preperation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):

        # Map class_id and names
        names[class_id] = fx[:-4]

        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        # Create labels for each image in the file
        target = class_id*np.ones((data_item.shape[0])) # Creates a labels array for all the faces in a file with class_ID
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)

# Concatenate both into single train dataset
trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

# Testing
# Initialize Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    ret, frame = cap.read()
    if ret ==False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.2, 4)

    for face in faces:
        x,y,w,h = face

        # Get the face Region of Interest
        offset = 15
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())

        # Display the predicted name on the screen
        predicted_name = names[int(out)]
        cv2.putText(frame, predicted_name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2 ,cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255,255), 2)
    cv2.imshow("Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()