# Retinal OCT Image Classification — CNV / DME / DRUSEN / NORMAL



---

## Türkçe

### Proje Hakkında

Bu proje, retinal OCT (Optik Koherens Tomografi) görüntülerini dört sınıfa ayıran bir derin öğrenme çalışmasıdır: **CNV** (Koroidal Neovaskülarizasyon), **DME** (Diyabetik Maküla Ödemi), **DRUSEN** ve **NORMAL**. Hem TensorFlow hem de PyTorch framework'leri kullanılarak altı farklı model karşılaştırılmıştır.

### Veri Seti

| Veri Seti | Kaynak |
|---|---|
| Kermany OCT 2017 | [paultimothymooney / kermany2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) |

```
Orijinal Dağılım:
  CNV    : 37.205 (train) + 242 (test)
  DME    : 11.348 (train) + 242 (test)
  DRUSEN :  8.616 (train) + 242 (test)
  NORMAL : 26.315 (train) + 242 (test)
  Toplam : 84.484 görüntü
```

Sınıf dengesizliğine karşı **class weight** yöntemi uygulanmıştır. Eğitim seti %80 / %10 / %10 oranında train, val ve test olarak bölünmüştür.

### Modeller

Aynı veri seti üzerinde TensorFlow ve PyTorch ile paralel olarak 6 model eğitilmiş ve karşılaştırılmıştır:

| Sıra | Model | Accuracy | F1-Score (weighted) |
|---|---|---|---|
| 1 | **CNN - PyTorch** | **%89.76** | **0.8985** |
| 2 | CNN - TensorFlow | %86.16 | 0.8587 |
| 3 | EfficientNetB0 (PyTorch) | %85.00 | 0.8534 |
| 4 | ResNet50 (PyTorch) | %81.91 | 0.8196 |
| 5 | ResNet50 (TF) | %79.34 | 0.7959 |
| 6 | EfficientNetB0 (TF) | %75.93 | 0.7592 |

> **En iyi model: CNN - PyTorch** — Test Accuracy = **0.8976**

### Yontemler

- **Goruntu boyutu:** 64x64 (CNN modelleri), 128x128 (Transfer Learning modelleri)
- **Veri Artirma (Augmentation):** Yatay cevirme, donme (+-15 derece), parlaklik ve kontrast ayari (ColorJitter)
- **Sinif Agirlandirma:** `compute_class_weight` ile dengesizlik giderilmistir.
- **Transfer Learning:** EfficientNetB0 ve ResNet50 ImageNet agirliklari ile yuklenip Fine-Tuning uygulanmistir.
- **Callbacks (TF):** `EarlyStopping`, `ReduceLROnPlateau`
- **Degerlendirme:** Confusion Matrix, Classification Report (precision / recall / F1 per class)

### Kurulum ve Calistirma

```bash
# Gereksinimleri yukleyin
pip install tensorflow torch torchvision scikit-learn pillow matplotlib seaborn pandas

# Notebook'u calistirin
jupyter notebook retina-o.ipynb
```

> **Not:** Veri seti Kaggle uzerinden otomatik cekilmektedir. Yerel calistirma icin `BASE_PATH` degiskenini kendi sisteminize gore guncelleyin.

### Cikti Dosyalari

```
best_cnn_tf.keras          → En iyi TF CNN modeli
best_cnn_torch.pth         → En iyi PyTorch CNN modeli
best_eff_tf.keras          → En iyi TF EfficientNetB0 modeli
best_eff_torch.pth         → En iyi PyTorch EfficientNetB0 modeli
best_resnet_tf.keras       → En iyi TF ResNet50 modeli
best_resnet_torch.pth      → En iyi PyTorch ResNet50 modeli
```

---

## English

### About the Project

This project is a deep learning study for classifying retinal OCT (Optical Coherence Tomography) images into four categories: **CNV** (Choroidal Neovascularization), **DME** (Diabetic Macular Edema), **DRUSEN**, and **NORMAL**. Six different models were trained and compared using both TensorFlow and PyTorch frameworks side by side.

### Dataset

| Dataset | Source |
|---|---|
| Kermany OCT 2017 | [paultimothymooney / kermany2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) |

```
Original Distribution:
  CNV    : 37,205 (train) + 242 (test)
  DME    : 11,348 (train) + 242 (test)
  DRUSEN :  8,616 (train) + 242 (test)
  NORMAL : 26,315 (train) + 242 (test)
  Total  : 84,484 images
```

Class imbalance was handled using **class weights**. The training set was further split into 80% / 10% / 10% for train, validation, and test.

### Models

Six models were trained in parallel using both TensorFlow and PyTorch on the same dataset:

| Rank | Model | Accuracy | F1-Score (weighted) |
|---|---|---|---|
| 1 | **CNN - PyTorch** | **89.76%** | **0.8985** |
| 2 | CNN - TensorFlow | 86.16% | 0.8587 |
| 3 | EfficientNetB0 (PyTorch) | 85.00% | 0.8534 |
| 4 | ResNet50 (PyTorch) | 81.91% | 0.8196 |
| 5 | ResNet50 (TF) | 79.34% | 0.7959 |
| 6 | EfficientNetB0 (TF) | 75.93% | 0.7592 |

> **Best model: CNN - PyTorch** — Test Accuracy = **0.8976**

### Methods

- **Image size:** 64x64 for CNN models, 128x128 for Transfer Learning models
- **Data Augmentation:** Horizontal flip, rotation (+-15 degrees), brightness and contrast adjustment (ColorJitter)
- **Class Weighting:** `compute_class_weight` used to address class imbalance
- **Transfer Learning:** EfficientNetB0 and ResNet50 loaded with ImageNet weights, followed by Fine-Tuning
- **Callbacks (TF):** `EarlyStopping`, `ReduceLROnPlateau`
- **Evaluation:** Confusion Matrix, Classification Report (precision / recall / F1 per class)

### Setup & Usage

```bash
# Install requirements
pip install tensorflow torch torchvision scikit-learn pillow matplotlib seaborn pandas

# Run the notebook
jupyter notebook retina-o.ipynb
```

> **Note:** The dataset is pulled automatically from Kaggle. For local execution, update the `BASE_PATH` variable to match your local directory structure.

### Output Files

```
best_cnn_tf.keras          → Best TF CNN model weights
best_cnn_torch.pth         → Best PyTorch CNN model weights
best_eff_tf.keras          → Best TF EfficientNetB0 model weights
best_eff_torch.pth         → Best PyTorch EfficientNetB0 model weights
best_resnet_tf.keras       → Best TF ResNet50 model weights
best_resnet_torch.pth      → Best PyTorch ResNet50 model weights
```

### Results Summary

The custom CNN trained with PyTorch achieved the best performance at **89.76% test accuracy**, surpassing all transfer learning models in this experimental setup. Notably, PyTorch implementations consistently outperformed their TensorFlow counterparts across all three architectures.

---

## License

This project is for educational and research purposes. Please refer to the original dataset license on Kaggle before any commercial use.
