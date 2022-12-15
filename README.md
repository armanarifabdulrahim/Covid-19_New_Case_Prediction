![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)


# Predicting the Spread of COVID-19 in Malaysia: A Machine Learning Approach

## Project description

This aim of this project is to predict the TREND of COVID-19 spread in Malaysia
These are steps used to complete this project.

#### 1. Import Necesary Modules

#### 2. Data Loading
    - Loaded csv dataset by using pandas load_csv()
    
#### 3. Data Inspection
    - Inspected the text data for the info, duplicates and missing data. 
    
#### 4. Data Cleaning
    - Filled null values 
![img](Resources/Cleaned.png)
    
#### 5. Data Preprocessing
    - Used Min Max Scaler to normalize data
    
#### 6. Model Development
    - Created a model using Tensorflow Long Short-Term Memory (LSTM) with 2 layers and 64 nodes for every layer with Dropout and 25 epochs.
    (*Do increase the nodes, layers and epoch for better results)
    - Used callbacks(ie. TensorBoard and Early Stopping) to prevent overfitting.
    
#### 7. Model Evaluation
    - Evaluated the model by using Mean Absolute Percentage Error (MAPE).
    
#### 8. Save the Model
    -  Saved the model
 
## Results

    - The model are able predict the COVID-19 trend.
![img](Resources/output.png)
 
    - The Mean Absolute Percentage Error (MAPE) is around 0.167%. This model did a great job of predicting COVID-19 spread trend.   
![img](Resources/MAPE2.PNG)

    - Tensorboard graph shows no overfitting or underfitting
![img](Resources/tb_loss.PNG)


## Acknowledgement
The dataset used for this project is by *[The Ministry of Health Malaysia (MOH)](https://github.com/MoH-Malaysia/covid19-public)*
 
