import kagglehub
from pathlib import Path
import csv
import cv2

def get_labels(i : int):
    path = kagglehub.dataset_download("iamtushara/face-detection-dataset")
    dataset_path = Path(path) / "merged"

    labels_validation = dataset_path / "labels" / "validation"
    img_path = dataset_path / "images" / "validation"

    idx = 0
    fieldsName = ['filename', 'c_min', 'c_max', 'r_min', 'r_max']
    with open('../CSVs/labels.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldsName)
        for txt in labels_validation.iterdir():
            if idx == i:
                break
            img_name = Path(txt).stem

            img_file = img_path / (img_name + ".jpg")
            img_file_jpeg = img_path / (img_name + ".jpeg")

            src = None
            if img_file.exists():
                src = cv2.imread(str(img_file))
            elif img_file_jpeg.exists():
                src = cv2.imread(str(img_file_jpeg))

            if src is None:
                print(f"img {img_name} fail!!!")
                continue

            if txt.suffix.lower() != '.txt':
                    continue
            idx += 1
            with open(txt, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    line = l.split() #x y w h
                    line.pop(0)
                    x_center_norm = float(line[0])
                    y_center_norm = float(line[1])
                    w_norm = float(line[2])
                    h_norm = float(line[3])
                    h,w,c = src.shape
                    x1 = int((x_center_norm * w) - (w_norm * w /2))
                    x2 = int((x_center_norm * w) + (w_norm * w /2))
                    y1 = int((y_center_norm * h) - (h_norm * h /2))
                    y2 = int((y_center_norm * h) + (h_norm * h /2))
                    writer.writerow([img_name, x1, x2, y1, y2])


