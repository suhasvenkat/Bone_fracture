🦴 Bone Fracture Detection Using CNN (ResNet-50)

An AI-powered system for detecting bone fractures from X-ray images using Deep Learning and a user-friendly GUI.

✨ Highlights

✅ Deep Learning–based fracture detection

✅ Transfer Learning with ResNet-50

✅ Binary Classification: Fracture / Normal

✅ Interactive GUI application

✅ Academic & research-ready

📌 Project Overview

Bone fracture detection from radiographic images is a crucial yet time-consuming task for medical professionals.
This project automates the detection process using a Convolutional Neural Network (CNN) built on ResNet-50, enabling fast and accurate classification of X-ray images.

A desktop GUI is included, allowing users to upload X-ray images and receive predictions instantly.

🧠 Model Details

Architecture: ResNet-50

Learning Type: Transfer Learning

Pretrained On: ImageNet

Task: Binary Classification

Output: Fracture / Normal

Loss Function: Binary Cross-Entropy

Optimizer: Adam

📂 Dataset Information

X-ray images of human bones

Two categories:

🟥 Fracture

🟩 Normal

Image size: 224 × 224

Dataset split:

Training set

Testing set

Data augmentation applied to reduce overfitting

Public datasets such as Kaggle or MURA-style datasets can be used.

⚙️ Tech Stack
Category	Tools
Language	Python
Deep Learning	TensorFlow, Keras
Image Processing	OpenCV, Pillow
GUI	Tkinter
Visualization	Matplotlib
Utilities	NumPy, Scikit-learn
🔁 Workflow

Load X-ray images

Preprocess & normalize images

Apply data augmentation

Train ResNet-50 model

Evaluate performance

Save trained model

Predict using GUI

🚀 Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/bone-fracture-detection.git
cd bone-fracture-detection

2️⃣ Install Dependencies
pip install -r requirements.txt


🔔 Recommended Python version: 3.8 – 3.10

▶️ Run the Application (GUI)

After installing the dependencies, start the GUI:

python mainGUI.py

🖥️ How to Use the GUI

Launch the application

Click Upload Image

Select an X-ray image

Click Predict / Detect Fracture

View the result:

✔️ Fracture

❌ Normal

(Optional) Prediction confidence displayed

📊 Model Performance
Metric	Result
Accuracy	~90%
Precision	High
Recall	High
F1-Score	Balanced

Results vary based on dataset size and quality.

📁 Project Structure
bone-fracture-detection/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── models/
│   └── resnet50_model.h5
│
├── train.py
├── test.py
├── predict.py
├── mainGUI.py
├── requirements.txt
└── README.md

🔮 Future Improvements

Multi-class fracture classification

Fracture localization using Grad-CAM

Web or mobile application deployment

Integration with hospital imaging systems
