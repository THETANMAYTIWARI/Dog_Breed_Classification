## 🐶 Dog Breed Classification

A deep learning pipeline (Colab + PyTorch + OpenCV) that 

(1) Detects humans via Haar cascades.

(2) Detects dogs using pretrained VGG16 (ImageNet dog classes).

(3) Classifies dog breeds (133 classes) using transfer learning.

### 🔗 Dataset (Drive): https://drive.google.com/drive/folders/1jwKZDJCx_3YeIIVbBml-a9ATzXpUTL1o?usp=sharing


### 🧠 Project Overview

Combine classical CV (face detection) and deep learning (VGG16 transfer learning) to detect whether an image contains a human or a dog and predict the dog breed. Includes a scratch CNN baseline + a transfer-learned VGG16 head for improved performance.

### ⚙️ Technologies Used

* Languages: Python
* Libraries / Tools: PyTorch, torchvision, OpenCV, Pillow, NumPy, Matplotlib, tqdm, Google Colab
* ML Concepts: Transfer learning, data augmentation, CrossEntropyLoss, SGD, TF preprocessing
* Dataset stats (used in notebook): ~13,233 human images, ~8,356 dog images (train/valid/test splits included)

### 📊 Methodology

🔹 Face Detection (OpenCV)

* Haar cascade (haarcascade_frontalface_alt.xml) to detect human faces.

* Observed: ~99% of sampled human images detected; ~6% false positives on dog images.

🔹 Dog Detection (VGG16 ImageNet)

* Use pretrained VGG16; image predicted index ∈ [151, 268] → dog class.

* Observed: ~95% detection rate on dog samples, ~0% humans misclassified as dogs in sample test.

🔹 CNN (from scratch) — baseline

* Small 3-conv network → FC → 133 outputs.

* Trained 20 epochs (SGD lr=0.01).

* Result: underfitting — ~10% test accuracy.

🔹 Transfer Learning (VGG16 head fine-tune)

* Freeze features layers; replace classifier's last layer with nn.Linear(in_features, 133).

* Train classifier head (SGD lr=0.001), strong augmentations (RandomResizedCrop, flip, rotation, normalization).

* Trained ~70 epochs; best validation loss ~0.97.

* Final test accuracy ~71% (solid baseline for 133 classes).

🔹 Prediction Pipeline

* detect(img_path) integrates face detector, dog detector, and breed predictor.

* User-friendly titles:

* dog & not human → “Hello, dog — predicted breed: …”

* human & not dog → “Hello, human — you look like …”

* both → combined friendly message

### 📈 Results Summary
| Model                     |                       Framework | Test Accuracy |
| ------------------------- | ------------------------------: | ------------: |
| CNN (scratch)             |                PyTorch (custom) |          ~10% |
| Transfer Learning (VGG16) | PyTorch (fine-tuned classifier) |      **~71%** |

### 📍 Key Visualizations / Outputs

* Sample predictions showing image + predicted breed title.

* Training & validation loss curves (per epoch).

* Confusion / misclassification analysis for hardest breeds.

### 🧩 Future Enhancements

* Try stronger backbones (ResNet50, EfficientNet) for higher accuracy.

* Unfreeze more features layers for deeper fine-tuning.

* Use class balancing, focal loss, or oversampling for rare breeds.

* Add model explainability (Grad-CAM) and a lightweight deployment (ONNX/TorchScript + Streamlit).
