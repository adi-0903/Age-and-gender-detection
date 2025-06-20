# 🎥 Gender Recognition and Age Estimation (Real-Time with Video Playback)

This project detects **faces** from a **webcam feed**, predicts **age** and **gender** using a pretrained **WideResNet model**, and plays a corresponding **video** based on the prediction. It's ideal for interactive installations, digital signage, or AI-based kiosks.

---

## 🧠 What It Does

- Uses **OpenCV** to capture video frames and detect faces.

- Uses a **pretrained WideResNet model** to estimate:

  - **Age** (0–100)

  - **Gender** (Male or Female)

- Plays a **video from a folder** depending on age/gender:

  - 👨 **Male (25–30)** → `25-30/`

  - 👨‍🦰 **Male (30–35)** → `30-35/`

  - 👩 **Female (25–30)** → `F25-30/`

---

## 📁 Folder Structure

Gender-Recognition-and-Age-Estimator/

│

├── pretrained_models/

│ ├── haarcascade_frontalface_alt.xml # OpenCV face detector

│ └── weights.18-4.06.hdf5 # Age/Gender pretrained model

│

├── 25-30/ # Videos for males 25–30

├── 30-35/ # Videos for males 30–35

├── F25-30/ # Videos for females 25–30

│

├── realtime_demo.py # Main Python script

├── wide_resnet.py # WideResNet model definition

├── README.md # You're reading it!

├── requirements.txt # Dependencies

## 🔧 Installation Guide

### 1. Clone the Repository

git clone https://github.com/yourusername/Gender-Recognition-and-Age-Estimator.git

cd Gender-Recognition-and-Age-Estimator


# 2. Set Up Python Environment (recommended)

conda create -n tf-env python=3.8

conda activate tf-env


# 3. Install Dependencies

pip install -r requirements.txt

# If you don’t have requirements.txt, install manually:

pip install opencv-python tensorflow keras numpy


## 📦 Required Files

# ✅ 1. Haar Cascade Face Detector

Download:

https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml

Save it in the folder:

pretrained_models/haarcascade_frontalface_alt.xml

# ✅ 2. Pretrained Age-Gender Model Weights

Download the model manually from this link:

📥 weights.18-4.06.hdf5 

https://drive.google.com/file/d/1rZ2ChR_RIeLxztyQ0DB2FfVHeBM7576b/view?usp=sharing

Save it here:

pretrained_models/weights.18-4.06.hdf5


# 💡 How It Works (Overview)

+---------------------+
| Webcam Live Feed    |
+---------------------+

        |
        
        v

+-------------------------+
| Face Detection (OpenCV) |
+-------------------------+
        |
        v
+-----------------------------+
| WideResNet Model (Keras)    |
|  - Predict Age & Gender     |
+-----------------------------+
        |
        v
+-----------------------------+
| If Age/Gender match group:  |
| Play relevant video         |
+-----------------------------+


# 📊 Model Details

Architecture: Wide Residual Network

Output:

Age: Regression from softmax over 101 age classes

Gender: Binary (Male or Female)

Training Dataset: IMDB-WIKI (500K+ labeled images)
