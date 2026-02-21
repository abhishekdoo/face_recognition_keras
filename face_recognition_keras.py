"""
Copyright 2026 Abhishek Mishra

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""

import os
import cv2
import numpy as np
import warnings
from mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = "dataset"
TEST_DIR = "test"
OUTPUT_DIR = "output"
THRESHOLD = 0.6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# INIT MODELS
# -----------------------------
detector = MTCNN()
embedder = FaceNet()

# -----------------------------
# FUNCTIONS
# -----------------------------
def detect_faces(image):
    try:
        faces = detector.detect_faces(image)
    except Exception:
        return []

    results = []
    for f in faces:
        x, y, w, h = f["box"]
        if w <= 0 or h <= 0:
            continue

        x, y = max(0, x), max(0, y)
        face = image[y:y+h, x:x+w]

        if face.size == 0:
            continue

        face = cv2.resize(face, (160, 160))
        results.append((face, (x, y, w, h)))

    return results


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


# -----------------------------
# BUILD FACE DATABASE
# -----------------------------
database = {}

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = detect_faces(img)
        if not faces:
            print(f"Skipped (no face): {img_path}")
            continue

        face_img, _ = faces[0]  # best face
        emb = embedder.embeddings(np.expand_dims(face_img, axis=0))[0]
        embeddings.append(emb)

    if embeddings:
        database[person] = np.mean(embeddings, axis=0)

print("Loaded persons:", list(database.keys()))

# -----------------------------
# PROCESS ALL TEST IMAGES
# -----------------------------
for test_img_name in os.listdir(TEST_DIR):
    test_img_path = os.path.join(TEST_DIR, test_img_name)

    if not test_img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image = cv2.imread(test_img_path)
    if image is None:
        continue

    faces = detect_faces(image)
    if not faces:
        print(f"No faces in {test_img_name}")
        continue

    for face_img, (x, y, w, h) in faces:
        emb = embedder.embeddings(np.expand_dims(face_img, axis=0))[0]

        name = "Unknown"
        best_score = 0.0

        for person, db_emb in database.items():
            score = cosine_similarity(emb, db_emb)
            if score > THRESHOLD and score > best_score:
                best_score = score
                name = person

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            image,
            f"{name} ({best_score:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    output_path = os.path.join(OUTPUT_DIR, test_img_name)
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")

print("✅ All test images processed.")
