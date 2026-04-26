# CIFAR-10 Classification with MobileNetV2

First ML model using **transfer learning** on CIFAR-10 (10 classes, 32x32 images).  
Uses **MobileNetV2** pretrained on ImageNet with **two-stage fine-tuning**:  
1. Base model frozen (20 epochs, LR=1e-3)  
2. Last 20 layers trainable (10 epochs, LR=1e-4) with data augmentation  

**Callbacks:** EarlyStopping (patience=5) & ReduceLROnPlateau  

**Results (final epoch):**  
- Training accuracy: 93.3%  
- Validation accuracy: 91.7%  
- Loss: 0.27  

Model saved as `cifar10.h5` for inference.

