<div align="center">
      <h1> <img src="https://github.com/Nobi004/Iris_Recognition_using_Deep_Learning/blob/main/Assets/Neutral%20Creative%20Professional%20LinkedIn%20Article%20Cover%20Image.png" width="700px"><br/> Iris Recognition using Deep Learning
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



![Data Details](https://github.com/Nobi004/Iris_Recognition_using_Deep_Learning/blob/main/Assets/Distribution%20of%20image%20sizes.png)

- **Image Quality and Variability:** Most of the image was high resulation having 800*600 shape.
- **Data Insights:** Some data are incorrect as inside one named folder one may find data of another folder. Dataset is highly imbalace.
  
![Distribution of Aspect Ratio](https://github.com/Nobi004/Iris_Recognition_using_Deep_Learning/blob/main/Assets/Distribution%20of%20aspect%20ratio.png)

#### 3.1.3 Data Preprocessing
Preparation of the dataset for modeling: Data set was turned into a pandas dataframe having label and image data. Data was balanced and
model was tested on both balanced and unbalanced data.
 As data was imbalance augmentation was slightly incresing the performance in the compensation of huge amount of time.
 
![Preprocessed Image](https://github.com/Nobi004/Iris_Recognition_using_Deep_Learning/blob/main/Assets/Preprocessed%20Image%20sample.png)

## 3.2 Model Engineering
Test was done on both custom model .
Mostly used Convolution and Maxpooling layer were used repetedly. Then in the second part dense and dropout layer used after flattening.

![Model History](https://github.com/Nobi004/Iris_Recognition_using_Deep_Learning/blob/main/Assets/Model%20Architecture.png)

### 3.2.1 Dataset Spliting
The dataset was divided into three subsets: Train dataset provided by kaggle divided into 2 set
- **Training Set:** The largest portion, used to train the model. 80% of the Train dataset provided by kaggle was used for training.
- **Validation Set:** Used to tune model parameters and prevent overfitting. 20% of the Train dataset provided by kaggle was used as validation dataset.
- Test dataset provided by kaggle was reserved for evaluating the model's performance on unseen data.

### 3.2.2 Model Architecture
**Layer Structure:**
- In the first section 7 Convolutional layer followed by 7 Maxpooling layer used , number of neurons were 32, 64 ,128,256 &  512. kernel size
were (5,5) .

```
def create_model():
    """
    Create the model architicure and compile it, call on pre-set values.
    Returns:
        model (keras.Sequential): a model compiled with its layers
    """
    padding = 'same'
    poolpadding = 'valid'

    model = Sequential([
        Input(input_shape),
        ####### Features extraction
        
        Conv2D(32, (5, 5), padding=padding, activation=activation, name="Conv1"),
        BatchNormalization(axis=-1, name="BN1"),  
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool1"),
        GaussianNoise(0.1, name="GaussianNoise"), 
        Dropout(0.1, name="Dropout1"),

        Conv2D(64, (5, 5), padding=padding, activation=activation, name="Conv2"),
        BatchNormalization(axis=-1, name="BN2"),  
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool2"),
        Dropout(0.1, name="Dropout2"),

        Conv2D(128, (5, 5), padding=padding, activation=activation, name="Conv3"),
        BatchNormalization(axis=-1, name="BN3"),  
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool3"),
        Dropout(0.25, name="Dropout3"),

        Conv2D(256, (3, 3), padding=padding, activation=activation, name="Conv4"),
        BatchNormalization(axis=-1, name="BN4"),  
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool4"),
        Dropout(0.25, name="Dropout4"),

        Conv2D(256, (3, 3), padding=padding, activation=activation, name="Conv5"),
        BatchNormalization(axis=-1, name="BN5"),  
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool5"),
        Dropout(0.25, name="Dropout5"),
        
        Conv2D(512, (3, 3), padding=padding, activation=activation, name="Conv6"),
        BatchNormalization(axis=-1, name="BN6"),  
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool6"),
        Dropout(0.45, name="Dropout6"),
        
        Conv2D(512, (2, 2), padding=padding, activation=activation, name="Conv7"),
        BatchNormalization(axis=-1, name="BN7"),  
        MaxPooling2D(pool_size=(2, 2), padding=poolpadding, name="Mpool7"),
        Dropout(0.5, name="Dropout7"),    
 ```


- In the second part 7 dense layer were used followed by dropout layer after flattening. In this case number of neurons or filters were
512,256, 128,64,32 and 10% to 30% dropout was done to prevent overfitting.
 ```
#### Flatten and fully connected layers, classifier using relu sofftmax
        Flatten(),
       # Dense(64, activation=activation, name = "Dense0"),
       # Dense(1024, activation=activation, name = "Dense6"),
        #
        Dropout(0.3, name="Dropout13"),
        
        Dense(512, activation=activation, name = "Dense3"),
        Dropout(0.1, name="Dropout11"),
        Dense(256,activation=activation,name = "Dense4"),
        Dropout(0.1, name="Dropout12"),
        Dense(128,activation=activation,name = "Dense5"),
        Dropout(0.15, name="Dropout8"),
        #Dense(128,activation=activation,name = "Dense6"),
        Dense(64, activation=activation, name = "Dense2"),
        Dropout(0.1, name="Dropout10"),
        
        Dense(32, activation=activation, name = "Dense1"),
        #Dropout(0.1, name="Dropout9"),
        
      
        Dense(2000, activation='softmax', name="SoftmaxClasses"),
    ],
    name="IRISRecognizer")
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model
```
- **Activation Functions:** Except the last layer where "Softmax" activation function was used in all other case "Leaky Relu" activation
function was used.
### 3.2.3 Model Training and Validation
Model was tranied on  balanced  data for 200 epoch.
- Training and validation accuracy by custom model on balanced dataset
- ![accuracy graph](https://github.com/Nobi004/Iris_Recognition_using_Deep_Learning/blob/main/Assets/Model%20Accuracy.png)
- Training and validation loss by custom model on balanced dataset
- ![Loss Learning Curve](https://github.com/Nobi004/Iris_Recognition_using_Deep_Learning/blob/main/Assets/Loss%20learning%20curve.png)
## 3.3 Evaluation and Analysis
- Validation and test accuracy by custom model on balanced dataset
- ![Evalution Code and Results](https://github.com/Nobi004/Iris_Recognition_using_Deep_Learning/blob/main/Assets/Evalution%20Code%20and%20Results.png)
### 3.3.1 Performance Testing
- Some predictions
- ![Some Output predictions](https://github.com/Nobi004/Iris_Recognition_using_Deep_Learning/blob/main/Assets/Some%20prediction%20outputs.png)

