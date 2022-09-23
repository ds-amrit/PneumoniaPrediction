# PneumoniaPrediction
GOAL: Given a lung x-ray detect wheather it has pneumonia or its healthy using a machine learning model. Finally, deployed it using Flask application.

In this POC computer vision and machine learning are used to distinguish lung x-rays.

The dataset downloaded from kaggle has 

Set: train   normal images: 1575  pneumonia images:3875

Set: val   normal images: 8  pneumonia images:8

Set: test   normal images: 234  pneumonia images:390

![image](https://user-images.githubusercontent.com/107737679/190056638-fc2f2692-70ba-4ac2-ab85-a7f5a23db3c0.png)


In the training dataset the target variable distribution looks like:

![image](https://user-images.githubusercontent.com/107737679/192058529-852ea7fc-ec09-404e-a2ff-f56bd7e35a4a.png)

The machine learning models used for prediction:

1. K- Neighbours Classifier: 
 
    Accuracy of KNN model(Train): 91.56%
    
    ![image](https://user-images.githubusercontent.com/107737679/192058981-fc217e64-28bd-4fa9-880a-fa5be20fd787.png)


2. Logistic Regression:
 
   Accuracy of Logistic Regression Model (Train) =93.54%
   Accuracy of Logistic Regression Model (Test) =88.14%
   
   ![image](https://user-images.githubusercontent.com/107737679/192059585-e3ababce-2c9b-4fc4-83a1-f05b5bb74ccc.png)

Finally, Logistic regression model is dumped to predict the Lung condition in the Flask app.





