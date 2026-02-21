# Face Recognition Environment Setup

## ✅ Step 1 --- Check Python Version

``` bash
python3.10 --version
```

------------------------------------------------------------------------

## ✅ Step 2 --- Create Virtual Environment

``` bash
python3.10 -m venv keras_env
```

------------------------------------------------------------------------

## ✅ Step 3 --- Activate Virtual Environment

``` bash
source keras_env/bin/activate
```

------------------------------------------------------------------------

## ✅ Step 4 --- Upgrade pip

``` bash
pip install --upgrade pip
```

------------------------------------------------------------------------

## ✅ Step 5 --- Install Dependencies

``` bash
pip install numpy
pip install scipy
pip install tensorflow==2.20.0
pip install opencv-python
pip install keras-facenet
pip install mtcnn
```

------------------------------------------------------------------------

## ✅ Step 6 --- Create Project Folder

``` bash
mkdir face_project
cd face_project
```

------------------------------------------------------------------------

## ✅ Step 7 --- Create Required Directories

``` bash
mkdir dataset
mkdir test
mkdir output
```

------------------------------------------------------------------------

## ✅ Step 8 --- Create Dataset Structure (Known Faces)

Example:

``` bash
mkdir dataset/Adi
mkdir dataset/Rahul
mkdir dataset/Mom
```

Place images inside:

    dataset/
        Adi/
            img1.jpg
            img2.jpg
        Rahul/
            img1.jpg
        Mom/
            img1.jpg

------------------------------------------------------------------------

## ✅ Step 9 --- Add Test Images

Place images directly inside:

    test/
        photo1.jpg
        group.jpg

------------------------------------------------------------------------

## ✅ Step 10 --- Run Face Recognition Script

``` bash
python face_recognition_keras.py
```
