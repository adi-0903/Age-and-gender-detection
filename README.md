# Gender-Recognition-and-Age-Estimator

This project uses a WideResNet deep learning model to estimate a person’s age and gender from webcam input in real time. Based on the detected age and gender, it plays a personalized video from a specified folder.

📸 #Features

Real-time face detection using OpenCV.

Age and gender prediction using a pretrained WideResNet.

Automatically plays videos for different demographics:

Male (25–30)

Male (30–35)

Female (25–30)

🛠️ Installation
Clone the repository

git clone https://github.com/yourusername/Gender-Recognition-and-Age-Estimator.git

cd Gender-Recognition-and-Age-Estimator

Create and activate a virtual environment

conda create -n tf-env python=3.8

conda activate tf-env


Install dependencies

pip install -r requirements.txt
⚙️ Required Files
Ensure the following files and folders exist:

Type	Location

Model	pretrained_models/weights.18-4.06.hdf5    [Link for this file(https://drive.google.com/file/d/1rZ2ChR_RIeLxztyQ0DB2FfVHeBM7576b/view?usp=sharing)]

Haarcascade	pretrained_models/haarcascade_frontalface_alt.xml

Videos	25-30/, 30-35/, F25-30/ (each with at least 1 video)
