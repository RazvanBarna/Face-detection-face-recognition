import joblib
import pandas
import cv2
import numpy as np

model = joblib.load("../TestPy/Models/model.pkl")
svm = model["svm"]
scaler = model["scaler"]
le = model["le"]
psi = model["psi"]
u = model["u"]
img = cv2.imread("test_face.png",cv2.IMREAD_GRAYSCALE)
#img_res = cv2.resize(img, (100, 100))
img_flat = img.flatten().astype(np.float32)
img_mean = img_flat - psi
img_pca = np.dot(img_mean, u)
img_scaled = scaler.transform([img_pca])
class_idx = svm.predict(img_scaled)
nume_persoana = le.inverse_transform(class_idx)
with open('prediction.txt','w') as f:
    f.write(nume_persoana[0])