import numpy as np
import cv2
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import train_test_split

path = Path("../TestPy/MyDataset/Faces/Faces/Faces/")
images = []
for p in sorted(path.iterdir()):
    img_aux = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img_aux is None:
        continue
    img_res = cv2.resize(img_aux, (100, 100))
    images.append(img_res.flatten())

Xd = np.array(images).astype(np.float32)

psi = np.mean(Xd, axis=0)
psi_img = psi.reshape(100, 100)

A = Xd - psi
C = np.dot(A,np.transpose(A))
vect_vals, vect_props = np.linalg.eigh(C)
idx = np.argsort(vect_vals)[::-1]
vect_vals = vect_vals[idx]
vect_props = vect_props[:,idx]

K = 150
new_V_props = vect_props[:,:K]
u = np.dot(np.transpose(A),new_V_props)
u = u / np.linalg.norm(u, axis=0)

w = np.dot(A,u)

df = pd.read_csv('../TestPy/MyDataset/Faces/Dataset.csv')
df = df.sort_values(by='id')
arr = df['label'].to_numpy()

le = LabelEncoder()
arr_encoded = le.fit_transform(arr)
scaler = StandardScaler()
w_scaled = scaler.fit_transform(w)

X_train, X_test, y_train, y_test = train_test_split(w_scaled, arr_encoded, test_size=0.25, random_state=10)
svm = SVC(random_state=10, probability=True)
svm.fit(X_train, y_train)
model_data = {
    "svm": svm,
    "scaler": scaler,
    "le": le,
    "psi": psi,
    "u": u,
}
with open("../TestPy/Models/model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Done training model")
#probabilities = svm.predict_proba(X_test)
#y_pred = svm.predict(X_test)