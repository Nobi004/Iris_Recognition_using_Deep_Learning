<div align="center">
      <h1> <img src="https://github.com/Nobi004/portfolio_nobi/blob/main/assets/Neutral%20Creative%20Professional%20LinkedIn%20Article%20Cover%20Image.png" width="200px"><br/> Iris Recognition using Deep Learning
 Documentation </h1>
     </div>

<body>
<p align="center">
  <a href="mailto:mdmnb435@gmail.com"><img src="https://img.shields.io/badge/Email-mdmnb435%40gmail.com-blue?style=flat-square&logo=gmail"></a>
  <a href="https://github.com/nobi004"><img src="https://img.shields.io/badge/GitHub-Mahmudun Nobi-lightgrey?style=flat-square&logo=github"></a>
  <a href="https://linkedin.com/in/nobi04"><img src="https://img.shields.io/badge/LinkedIn-Mahmudun%20Nobi-blue?style=flat-square&logo=linkedin"></a>
  <a href="https://mahmudunnobi.streamlit.app/"><img src="https://img.shields.io/badge/Website-Mahmudun%20Nobi-lightgrey?style=flat-square&logo=google-chrome"></a>

  <br>
  <img src="https://img.shields.io/badge/Phone-%2B8801530045859-green?style=flat-square&logo=whatsapp">
  <a href="https://www.hackerrank.com/profile/mdmnb435"><img src="https://img.shields.io/badge/Hackerrank-Mahmudun%20Nobi-green?style=flat-square&logo=hackerrank"></a>
</p>




# Iris Recognition using Deep Learning


## Table of Contents
1. [Introduction](#introduction)
   - 1.1 Program Overview
   - 1.2 Challenge Objective
2. [Data Source](#data-source)
   - 2.1 Dataset Overview
   - 2.2 Accessing the Dataset
3. [Task Specifications](#task-specifications)
   - 3.1 Data Management
     - 3.1.1 Data Acquisition
     - 3.1.2 Exploratory Data Analysis (EDA)
     - 3.1.3 Data Preprocessing
   - 3.2 Model Engineering
     - 3.2.1 Dataset Splitting
     - 3.2.2 Model Architecture
     - 3.2.3 Model Training and Validation
   - 3.3 Evaluation and Analysis
     - 3.3.1 Performance Testing
     - 3.3.2 Metrics Reporting
   - 3.4 Conclusion and Future Work
## 1. Introduction

### 1.1 Program Overview
Iris recognition is a biometric method used to identify individuals based on the unique patterns in the iris of their eyes. This project aims to develop an efficient and accurate iris recognition system using deep learning techniques. By leveraging modern machine learning architectures such as Convolutional Neural Networks (CNNs), we aim to improve the accuracy of identifying individuals from iris images.
## 2. Data Source

### 2.1 Dataset Overview
The CASIA-Iris-Thousand dataset is one of the most widely used iris image datasets for biometric identification. It contains high-quality iris images captured under controlled conditions, making it ideal for training and evaluating iris recognition models. This dataset has been instrumental in research and development within the field of biometric recognition, specifically focusing on the unique patterns found in the human iris.

### 2.2 Accessing the Dataset
The dataset is hosted on Kaggle, a popular platform for data science competitions and collaborative projects.

## 3. Task Specifications

### 3.1 Data Management

#### 3.1.1 Data Acquisition
Data was downloaded from Kaggle

#### 3.1.2 Exploratory Data Analysis(EDA)
Dataset Structure:
   - **Number of Subjects:** The dataset contains images from 1,000 subjects.
   - **'Total Images': There are 20,000 iris images, with each subject contributing images from both the left and right eye.**
   - **'Resolution': The images have a consistent resolution, typically suitable for deep learning models without significant preprocessing.**
   - **'File Format': The images are stored in .bmp format.**



![Data Details](https://github.com/ArtificialIntelligenceResearch/Cassava-leaf-disease-classification-by-M.Nobi-/blob/main/Images/Screenshot%202024-07-15%20130344.png)

- **Image Quality and Variability:** Most of the image was high resulation having 800*600 shape.
- **Data Insights:** Some data are incorrect as inside one named folder one may find data of another folder. Dataset is highly imbalace.
- Most of them are Cassava Mosaic Disease (CMD).
![Some Data Sample](https://github.com/ArtificialIntelligenceResearch/Cassava-leaf-disease-classification-by-M.Nobi-/blob/main/Images/Screenshot%202024-07-15%20200508.png)

#### 3.1.3 Data Preprocessing
Preparation of the dataset for modeling: Data set was turned into a pandas dataframe having label and image data. Data was balanced and
model was tested on both balanced and unbalanced data.
 As data was imbalance augmentation was slightly incresing the performance in the compensation of huge amount of time.

## 3.2 Model Engineering
Test was done on both custom model .
Mostly used Convolution and Maxpooling layer were used repetedly. Then in the second part dense and dropout layer used after flattening.

![Model History](https://github.com/ArtificialIntelligenceResearch/Cassava-leaf-disease-classification-by-M.Nobi-/blob/main/Images/Screenshot%202024-07-15%20203319.png)

### 3.2.1 Dataset Spliting
The dataset was divided into three subsets: Train dataset provided by kaggle divided into 2 set
- **Training Set:** The largest portion, used to train the model. 80% of the Train dataset provided by kaggle was used for training.
- **Validation Set:** Used to tune model parameters and prevent overfitting. 20% of the Train dataset provided by kaggle was used as validation dataset.
- Test dataset provided by kaggle was reserved for evaluating the model's performance on unseen data.

### 3.2.2 Model Architecture
**Layer Structure:**
- In the first section 7 Convolutional layer followed by 7 Maxpooling layer used , number of neurons were 32, 64 & 128. kernel size
were (3,3) .

```
input_shape = (256,256,3)

inputs = Input(shape=input_shape)

model = augment(inputs)
model = Conv2D(32,(3,3),activation = 'relu')(model)
model = MaxPooling2D((2,2))(model)
model = Conv2D(64,(3,3),activation = 'relu')(model)
model = MaxPooling2D((2,2))(model)
model = Conv2D(64,(3,3),activation = 'relu')(model)
model = MaxPooling2D((2,2))(model)
model = Conv2D(128,(3,3),activation = 'relu')(model)
model = MaxPooling2D((2,2))(model)
model = Conv2D(128,(3,3),activation = 'relu')(model)
model = MaxPooling2D((2,2))(model)       
 ```


- In the second part 2 dense layer were used followed by dropout layer after flattening. In this case number of neurons or filters were
256, 128 and 35% and 30% dropout was done to prevent overfitting.
 ```
 model = Flatten()(model)
model = Dense(512,activation='relu')(model)
model = Dense(256,activation='relu')(model)
model = Dense(128,activation='relu')(model)
model = Dense(64,activation='relu')(model)
model = Dense(32,activation='relu')(model)
model = Dropout(0.2)(model)
outputs = Dense(5,activation='softmax')(model)
```
- **Activation Functions:** Except the last layer where "Softmax" activation function was used in all other case "Relu" activation
function was used.
- **Transfer Learning:** Pretrained EfficientNetB0 model was used to compare the performance.
### 3.2.3 Model Training and Validation
Model was tranied on both balanced and unbalanced data for 50 epoch.
- Training and validation accuracy by custom model on balanced dataset
- ![accuracy graph](https://github.com/ArtificialIntelligenceResearch/Cassava-leaf-disease-classification-by-M.Nobi-/blob/main/Images/Screenshot%202024-07-16%20124624.png)
- Training and validation accuracy by custom model on un-balanced dataset
- ![loss graph](https://github.com/ArtificialIntelligenceResearch/Cassava-leaf-disease-classification-by-M.Nobi-/blob/main/Images/Screenshot%202024-07-16%20125049.png)
## 3.3 Evaluation and Analysis
- Validation and test accuracy by custom model on balanced dataset
- ![evalution code and results](https://github.com/ArtificialIntelligenceResearch/Cassava-leaf-disease-classification-by-M.Nobi-/blob/main/Images/Screenshot%202024-07-16%20125555.png)
### 3.3.1 Performance Testing
- Confusion Matrix by custom model on balanced dataset
- ![confusion matrix](https://github.com/ArtificialIntelligenceResearch/Cassava-leaf-disease-classification-by-M.Nobi-/blob/main/Images/Screenshot%202024-07-16%20130020.png)
- image prediction by custom model trained on balanced dataset
