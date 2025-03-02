# Garbage Classification Using Deep Learning

## Overview
This project explores the impact of deep learning-based models on waste management processes. Accurate classification of waste plays a crucial role in achieving sustainability and economic development goals. To address inefficiencies in traditional methods, modern deep learning architectures such as **Vision Transformer (ViT) - Base, Swin Transformer-Tiny, InceptionV3, VGG19, and ResNet50V2** were used to solve a **10-class waste classification problem**.

## Dataset
- **Classes:** The dataset consists of images categorized into 10 different waste types:
  - **Cardboard**
  - **Trash**
  - **Biological**
  - **Paper**
  - **Plastic**
  - **Metal**
  - **Clothes**
  - **Shoes**
  - **Glass**
  - **Battery**

- **Dataset Preparation:**
  - The dataset is structured into subfolders where each folder represents a waste category.
  - Image preprocessing involves resizing images to **224x224** resolution and applying **data augmentation techniques**.

## Environment Setup
- Required dependencies:
  ```bash
  pip install tensorflow transformers torch torchvision scikit-learn numpy pandas matplotlib
  ```
- The models are implemented using **TensorFlow** and **PyTorch**.

## Models Used
### Transformer-Based Architectures
- **Vision Transformer (ViT-Base)**
- **Swin Transformer-Tiny**

### CNN-Based Architectures
- **InceptionV3**
- **ResNet50V2**
- **VGG19**

## Model Training
- The dataset is split into **80% training** and **20% validation**.
- The models are trained using **cross-entropy loss** and optimized using **Adam optimizer**.
- Data augmentation (rotation, flipping, normalization) is applied to improve generalization.
- **Early Stopping** and **Learning Rate Reduction on Plateau** are used to enhance training efficiency.

### Example: Training a ViT Model
```python
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=10)
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

def preprocess_images(example):
    image = Image.open(example['image_path']).convert("RGB")
    example['pixel_values'] = feature_extractor(images=image, return_tensors="pt").pixel_values[0]
    return example
```

## Performance Evaluation
- **Evaluation Metrics:**
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
  - **Confusion Matrix**

### Results
| Model                 | Accuracy (%) | F1-Score (%) |
|-----------------------|-------------|-------------|
| ViT-Base             | **95.55**    | **95.55**    |
| Swin Transformer-Tiny| **96.05**    | **96.05**    |
| InceptionV3          | 93.06        | 92.80       |
| ResNet50V2           | 91.96        | 91.50       |
| VGG19                | 85.78        | 85.30       |

- **Key Findings:**
  - **ViT-Base** achieved the highest accuracy of **95.55%**, demonstrating the effectiveness of transformer-based architectures in waste classification.
  - **Swin Transformer-Tiny** slightly outperformed ViT in accuracy (**96.05%**) but had similar F1-Score.
  - Among CNN models, **InceptionV3** performed best with **93.06% accuracy**.
  - **VGG19 had the lowest accuracy (85.78%)**, indicating its limitations in handling complex waste images.

## Conclusion
This study highlights the potential of deep learning models, particularly **transformers**, in improving waste classification processes. The results suggest that modern architectures can significantly enhance accuracy in real-world waste management systems.

### Future Work
- Expanding the dataset with more diverse waste categories.
- Optimizing models for real-time waste classification applications.
- Exploring lightweight transformer architectures for mobile deployment.

This project demonstrates the effectiveness of AI-driven waste classification and provides a foundation for future advancements in sustainable waste management.

