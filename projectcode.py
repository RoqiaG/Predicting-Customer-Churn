#import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import tensorflow as tf
#from tensorflow import keras

df=pd.read_csv("D:/scienceProject/Churn_Modelling.csv")
df
# to explor the data
df.info()
# Preprocessing
df["CustomerId"].nunique()
df["Surname"].nunique()
df.drop(columns=["Surname","CustomerId","RowNumber"],inplace=True)
df
#includes to encode the columns
le=LabelEncoder()
df["Gender"]=le.fit_transform(df.Gender)

df["Geography"].nunique()
df["Geography"]=le.fit_transform(df.Geography)

# split the label from the features
X = df.drop('Exited', axis=1)
y = df['Exited']
# split the data to train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#apply normalization to trainset & testset
#scaler = StandardScaler()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#y_train_categorical = to_categorical(y_train)
#y_test_categorical = to_categorical(y_test)

#preTrained model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100) # the number of decision trees 
model.fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_test, y_test)

# preTrained model svc
from sklearn.svm import SVC # support vector classifier 
model1 = SVC()
model1.fit(X_train, y_train)
model1.score(X_test, y_test)
model1.score(X_test, y_test)

# preTrained model XGB
import xgboost as xgb
from xgboost import XGBClassifier
model2 = XGBClassifier()
model2.fit(X_train, y_train)
model2.score(X_train, y_train)
model2.score(X_test, y_test)

# build ANN using keras
model = tf.keras.models.Sequential([

            # The first layers must specify the input shape always
            tf.keras.layers.Dense(10 , input_shape=(10,) ,activation='relu'),
            #tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(6, activation='relu'),
            #tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='relu'),
           # tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(1 ,activation='sigmoid')

])



model.compile( optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#from tensorflow.keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
#model.fit(X_train,y_train,epochs=100,batch_size=32 ,callbacks=[early_stopping])
model.fit(X_train, y_train, epochs=100, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5)
# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# accuracy_score(y_test,y_pred)