
# Iris Recognition using Deep Learning

This repository contains code and resources for building a deep learning model for iris recognition, a biometric identification technique. The project leverages convolutional neural networks (CNNs) and other machine learning methods to process and classify iris images.

## Key Features:
- **Data Handling**: Loading and preprocessing of large datasets (e.g., CASIA-Iris-Thousand dataset) for iris image classification.
- **Model Building**: Implementation of CNN architectures using TensorFlow/Keras with layers such as Conv2D, BatchNormalization, MaxPooling, and Dense.
- **Evaluation Metrics**: Includes precision, recall, accuracy, AUROC, and confusion matrices for model evaluation.
- **Optimization**: Use of early stopping, learning rate reduction, and other callbacks to optimize training performance.
- **Visualization**: Tools for visualizing image samples and confusion matrices.

## Requirements:
- Python, TensorFlow/Keras, PyTorch, OpenCV, NumPy, Pandas, Matplotlib, Seaborn

## How to Run:
1. Clone the repository:
   ```bash
   git clone https://github.com/Nobi004/Iris_Recognition_using_Deep_Learning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook iris-recognition.ipynb
   ```

## Dataset:
This project uses the [CASIA-Iris-Thousand](http://www.cbsr.ia.ac.cn/english/IrisDatabase.asp) dataset for training and evaluation.

## My Kaggle Notebook:
https://www.kaggle.com/code/mdmahmudunnobi/iris-recognition

## Acknowledgements:
Special thanks to the creators of the CASIA-Iris-Thousand dataset and the open-source libraries used in this project.

Feel free to explore the code and adapt it for your iris recognition needs.
