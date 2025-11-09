# **Sai Uday's Content**
## **Welcome to Day 4 of the Bootcamp!!!**
Today, we'll be learning how **image recognition** happens, and how **Deep Learning (DL)** works!

---

### **What are ANN's?**
An **Artificial Neural Network (ANN)** is a type of model in Deep Learning that tries to work like the human brain — it learns by finding patterns in data.

- Just like our brain has neurons that send signals to each other, an ANN has **artificial neurons (nodes)** connected in layers that pass information forward and adjust themselves to learn.
- ANN is made up of layers of neurons:
  - **Input Layer**: Receives the data.
  - **Hidden Layers**: Where the actual learning happens — the model adjusts its weights and biases for each neuron during training.
  - **Output Layer**: Produces the final prediction.

![Example of an ANN](../assets/ANN.png)

```python
# Define the model
model = Sequential([
    Input(shape=(4,)),            # Input layer (4 features)
    Dense(4, activation='relu'),  # Hidden layer 1
    Dense(4, activation='relu'),  # Hidden layer 2
    Dense(3, activation='sigmoid') # Output layer (multi-class classification)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

#### **Key Terms:**
- **Activation Function**: Adds non-linearity (mathematically) for a neuron, so the model can learn complex patterns instead of simple linear ones.
- **Optimizers**: Algorithms that improve model learning by adjusting weights and biases during training to reduce error.
- **Loss Function**: Measures how wrong the model’s predictions are — the model tries to minimize this loss while learning.

---

### **What are CNN's?**
A **Convolutional Neural Network (CNN)** is a type of Deep Learning model specifically designed to handle **grid-like data**, such as images.

- Instead of looking at the whole image at once, CNNs look at **small local regions (patches)** to learn patterns like edges, textures, shapes, and then combine them to recognize higher-level features.
- **Applications**: Image classification, object detection, face recognition, medical imaging, etc.

#### **How CNNs work?**
A CNN has **three main types of layers**:

##### Convolution Layer:
  - Responsible for finding features in the data, such as edges, shapes, and textures using **strides** and **kernels**.
##### Pooling Layer:
  - Reduces the size of the feature maps to make computation easier while retaining the most important features.
##### Fully Connected Layer:
  - Predicts the outcome based on the extracted features passed to it.

![Example of a CNN](../assets/cnn.png)
![Intuition](../assets/Intuition.png)

---

### **Basic Evolution of CNN**

#### **LeNet (1998):**
- One of the earliest CNNs, designed for handwritten digit recognition (e.g., MNIST dataset).
- Introduced **convolution** and **pooling layers**.

#### **AlexNet (2012):**
- Popularized deep CNNs by winning the ImageNet competition.
- Used **ReLU activation**, **dropout**, and **GPU training**.

#### **VGGNet (2014):**
- Used very deep networks with small (3×3) convolution filters.
- Demonstrated that **increasing depth improves performance**.

#### **GoogLeNet/Inception (2014):**
- Introduced **inception modules** for multi-scale feature extraction.
- Used **global average pooling** instead of fully connected layers.

#### **ResNet (2015):**
- Introduced **residual connections (skip connections)** to solve vanishing gradient problems.
- Enabled training of very deep networks (50+ layers).

---

### **Differences between an ANN and CNN**

| **Aspect**            | **ANN (Artificial Neural Network)** | **CNN (Convolutional Neural Network)** |
|------------------------|-------------------------------------|-----------------------------------------|
| **Main Idea**          | Fully connected layers that learn global patterns | Uses convolutional filters to learn local spatial patterns |
| **Typical Input**      | 1D feature vectors (tabular data)  | 2D/3D grid data (images, videos, volumes) |
| **Connectivity**       | Dense connections between neurons  | Local connectivity (receptive fields) + sparse connections |
| **Parameter Sharing**  | No (each weight is unique)         | Yes (same filter applied across spatial locations) |
| **Feature Learning**   | Learns global relationships        | Learns hierarchical features (edges → textures → objects) |
| **Translation Invariance** | Limited                       | Stronger (due to convolutions and pooling) |
| **Common Layers**      | Dense (fully connected), activation | Convolution, pooling, batch norm, fully connected |
| **Typical Use Cases**  | Tabular data, simple classification/regression | Image classification, object detection, segmentation |

---

### **Different Famous CNN Architectures (Concise Specs)**

#### **LeNet-5 (1998):**
- **Input**: 32×32 grayscale
- **Convs**: 3 conv layers (C1: 6 filters, C3: 16 filters, C5: 120 filters) + pooling layers
- **Fully Connected**: 2 (F6 and output)
- **Total Params**: ~60k
- **Notes**: Designed for MNIST; simple, small model for digit recognition.
![LeNet-5 architecture](../assets/lenet.png)

#### **AlexNet (2012):**
- **Input**: 224×224 RGB (original used 227×227)
- **Convs**: 5 conv layers + max-pooling layers
- **Fully Connected**: 3 FC layers
- **Total Params**: ~60M
- **Notes**: ReLU, dropout, GPU training, large kernels in early layers (11×11, 5×5).
![AlexNet architecture](../assets/alexnet.png)

#### **VGG (2014) — e.g., VGG-16 / VGG-19:**
- **Input**: 224×224 RGB
- **Convs**: VGG-16 = 13 conv layers (stacked 3×3 filters) + 5 max-pool layers; VGG-19 = 16 conv
- **Fully Connected**: 3 FC layers
- **Total Params**: VGG-16 ≈ 138M
- **Notes**: Very deep with uniform 3×3 filters; high parameter count.
![VGG architecture](../assets/vgg.png)

#### **ResNet (2015):**
- **Input**: 224×224 RGB
- **Convs**: ResNet-50 uses bottleneck blocks totaling 49 conv layers + 1 FC (counted as 50); ResNet-101 deeper
- **Fully Connected**: 1 final FC (after global pooling)
- **Total Params**: ResNet-50 ≈ 25M
- **Notes**: Residual (skip) connections that enable very deep networks and ease training.
![ResNet-50 architecture](../assets/resnet50.png)

---

### **Advantages of CNN**
- **Learns hierarchical spatial features** (edges → textures → objects).  
- **Parameter sharing & local connectivity** → fewer parameters and efficient learning for images.  
- **State-of-the-art** for vision tasks and has efficient variants for edge/mobile.

### **Disadvantages of CNN**
- **Data hungry** — needs large labeled datasets for best performance.  
- **Compute and memory intensive** (training often requires GPUs/TPUs).  
- **Can overfit** on small datasets.  

---

### **Hosting on Hugging Face**

#### **Prerequisites**
- **Hugging Face account**: [Sign up here](https://huggingface.co/)
- **Git and Git LFS installed**:
  - Windows: `git lfs install`
- **Python packages**:
  ```bash
  pip install huggingface_hub torch torchvision
  ```
- Train your CNN locally and save artifacts (weights, optional config, and helper code).

#### **Steps to Host:**
1. **Save model artifacts** (e.g., `pytorch_model.bin`, `config.json`, `model.py`).
2. **Create a repo** on Hugging Face (via CLI or web).
3. **Push files** using Git + LFS or Python API.
4. **Add a README** (model card) with usage instructions.
5. **Enable Inference API** (optional).

---

### **Notes / Tips**
- Use **Git LFS** for large weight files (>10 MB).
- Provide a **model.py** file to help others reconstruct the architecture.
- Add a **README.md** with usage, dataset, license, and metrics.
- Enable the **Inference API** for remote inference directly on Hugging Face.