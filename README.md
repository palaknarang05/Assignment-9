# Cats vs Dogs Image Classification using CNN (Assignment 9)

## Objective
Build a Convolutional Neural Network (CNN) to automate classification of pet
images into **Cats** and **Dogs**, for an animal welfare organization use case.

## Developer Info
**Name:** Palak Narang
**Registration Number:** 23BCE11819
**Application Number:** IN26011657
**Batch Number:** 1A

## Dataset
[Cats vs Dogs Classification Dataset — Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

Not included in this repository (per assignment instructions — Kaggle license
does not permit redistribution). Place downloaded images as:
```
data/
├── Cat/
│   ├── 0.jpg
│   └── ...
└── Dog/
    ├── 0.jpg
    └── ...
```
This run used a 1,399-image subset (700 cats / 699 dogs) to keep training
time reasonable on CPU; results below reflect that subset.

## Libraries Used
TensorFlow / Keras, NumPy, Matplotlib, Pillow (PIL), scikit-learn

## Methodology
1. **Data Understanding:** loaded dataset, inspected folder structure, displayed
   sample images with labels, counted classes/images, checked image dimensions.
2. **Preprocessing:** resized all images to 128×128, normalized pixel values to
   [0, 1], split 80/20 train/validation using Keras `ImageDataGenerator`.
3. **Model Development:** built and trained the CNN below for 10 epochs with
   Adam optimizer and binary crossentropy loss.
4. **Evaluation:** computed test accuracy, precision, recall, F1-score,
   confusion matrix, and accuracy/loss curves across epochs.

## CNN Architecture
```
Conv2D(32, 3x3, ReLU) → MaxPooling2D(2x2)
Conv2D(64, 3x3, ReLU) → MaxPooling2D(2x2)
Conv2D(128, 3x3, ReLU) → MaxPooling2D(2x2)
Flatten
Dense(128, ReLU)
Dense(1, Sigmoid)

Optimizer: Adam | Loss: Binary Crossentropy | Metric: Accuracy | Epochs: 10
Total params: 3,304,769
```

## Results (actual run)

**Dataset:** 1,399 images total → 1,120 train / 279 validation

| Metric | Value |
|---|---|
| Test Accuracy | **67.03%** |
| Test Loss | 0.7786 |
| Precision (Cat / Dog) | 0.69 / 0.65 |
| Recall (Cat / Dog) | 0.61 / 0.73 |
| F1-score (Cat / Dog) | 0.65 / 0.69 |

Confusion matrix, accuracy/loss curves, and sample images are generated
directly in `Assignment-9.ipynb` (all cells executed, outputs saved inline).

### Observations
1. Training accuracy climbed from ~48% (epoch 1) to ~90% (epoch 10), while
   validation accuracy plateaued around 63-68% with validation loss rising
   after epoch 3 — clear overfitting on this small (~1,120 image) training set.
2. Final test accuracy (67.03%) is well above the 50% random baseline but
   below what this architecture typically reaches on the full 25,000-image
   Kaggle set — dataset size, not architecture, was the limiting factor.
3. Precision/recall were reasonably balanced (Cat: 0.69/0.61, Dog: 0.65/0.73),
   with a slight bias toward predicting "Dog."
4. The train/test accuracy gap (89.9% vs 67.0%) confirms overfitting;
   dropout, batch normalization, and data augmentation would likely close
   this gap if scaled to the full dataset.

## Conclusion
This project built a CNN to classify cat and dog images, reaching 67.03% test
accuracy on a 1,399-image subset after 10 epochs. Stacked Conv2D +
MaxPooling2D blocks progressively extract low-level (edges, textures) and
higher-level (shapes, fur patterns) features while pooling reduces spatial
dimensions and adds translation invariance. Compared to a plain ANN, a CNN's
convolutional weight-sharing exploits the local spatial structure of images,
recognizing features like ears or paws regardless of position, with far
fewer parameters than a fully-connected network on raw pixels would require.
A limitation observed directly here is that CNNs need substantial labeled
data to generalize well — with only ~1,100 training images the model
overfit noticeably (89.9% train vs 67.0% test accuracy), and would benefit
from a larger dataset, dropout, and augmentation to close that gap.
