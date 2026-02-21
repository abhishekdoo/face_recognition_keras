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
import shutil
import numpy as np
import warnings
from mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
INITIAL_DATASET_DIR = "dataset"
TEST_DIR = "test"
THRESHOLD = 0.6
ITERATIONS = 10

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
        results.append(face)

    return results


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def build_database(dataset_dir):
    database = {}

    for person in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person)

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
                continue

            emb = embedder.embeddings(np.expand_dims(faces[0], axis=0))[0]
            embeddings.append(emb)

        if embeddings:
            database[person] = np.mean(embeddings, axis=0)

    return database


# -----------------------------
# ITERATIVE LOOP
# -----------------------------
current_dataset_dir = INITIAL_DATASET_DIR

for i in range(2, ITERATIONS + 2):

    output_dir = f"dataset{i}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🚀 Iteration {i-1}")
    print(f"Database → {current_dataset_dir}")
    print(f"Predict  → {TEST_DIR}")
    print(f"Output   → {output_dir}")

    # Build evolving database
    database = build_database(current_dataset_dir)

    print("Persons:", list(database.keys()))

    for img_name in os.listdir(TEST_DIR):

        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(TEST_DIR, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        faces = detect_faces(image)

        if not faces:
            print(f"No face: {img_name}")
            continue

        emb = embedder.embeddings(np.expand_dims(faces[0], axis=0))[0]

        name = "Unknown"
        best_score = 0.0

        for person, db_emb in database.items():
            score = cosine_similarity(emb, db_emb)

            if score > THRESHOLD and score > best_score:
                best_score = score
                name = person

        print(f"{img_name} → {name} ({best_score:.2f})")

        person_folder = os.path.join(output_dir, name)
        os.makedirs(person_folder, exist_ok=True)

        destination_path = os.path.join(person_folder, img_name)
        shutil.copy(img_path, destination_path)

    # Update dataset for next iteration
    current_dataset_dir = output_dir

print("\n✅ 10 Iterations Complete.")