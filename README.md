# LinguaSign AI: Real-Time Sign Language Translator 🤟

## 👥 The Team: BorNEO HackWknd 2026
* **Vickson Ferrel anak Raymond Lenggu** 
* **Muhammad Izzat bin Patihie**
* **Alvin Bong Kian Ting**
* **Romeo Ryan Munan anak Roland Munan**

---

## ❗ CRITICAL: Setup Requirement
**You MUST unzip the `Hackathon_Demo_Data.zip` file into the main directory.** The application expects a folder named `Hackathon_Demo_Data` to be present. This folder contains the sequence data required to initialize the vocabulary and mapping for the translation engine.

---

## 🚀 The Mission
**LinguaSign AI** is a functional Deep Learning prototype built for the **BorNEO HackWknd 2026**. Our mission is to bridge communication gaps by utilizing computer vision and high-performance Neural Networks to translate Malaysian Sign Language (BIM) into text in real-time.

### 🧠 Technical Architecture
* **Holistic Tracking:** Powered by **MediaPipe**, the system captures 543 landmarks (hands, pose, and face) at 30 FPS to ensure fluid movement tracking.
* **Normalized Data Extraction:** Keypoints are mathematically anchored to the base of the neck (Landmark 0). This normalization ensures the AI remains accurate regardless of the user's distance or position relative to the camera.
* **LSTM Neural Network:** We utilize a **Long Short-Term Memory (LSTM)** network. Unlike standard AI, LSTM analyzes the *temporal sequence* of movements, allowing the system to understand the "flow" and "intent" of a sign rather than just a static pose.

---

## 🛠️ Quick Start
### 1. Install Dependencies
Ensure you are using Python 3.12 or 3.13 within a virtual environment. Run:
```powershell
pip install -r requirements.txt
```

### 2. Launch the Engine
Run the main script to open the live camera feed:
```powershell
python live_translator_pro.py
```

## 🤖 AI Disclosure
In compliance with the **BorNEO HackWknd 2026** regulations, we disclose that Generative AI was utilized during this project as a collaborative partner for:

* **Environment Configuration:** Resolving complex dependency conflicts between TensorFlow, Protobuf, and NumPy for the Python 3.13 runtime.
* **Technical Debugging:** Optimizing real-time keypoint extraction and data anchoring logic.
* **Project Documentation:** Assisting in the clear communication of technical features within this README and the final project report.
---
## 📄 Project Deliverables
* **Project Report:** [Click here to view the PDF Report](./LinguaSign_Project_Report.pdf)
* **Demo Video:** [Watch the 8-Minute Walkthrough on YouTube](https://www.youtube.com/watch?v=Rf8-n1rsZ8c)

---

## 🏗️ Technical Breakdown for Judges
Our solution utilizes an **LSTM (Long Short-Term Memory)** architecture because sign language isn't just about static shapes—it's about the motion over time. By capturing 30 frames per sign, our model learns the temporal "signature" of each word, leading to much higher accuracy than traditional image classification.


