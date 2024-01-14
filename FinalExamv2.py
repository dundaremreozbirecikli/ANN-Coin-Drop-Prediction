#-------------------------------------------------------
# 
# Coin Drop Prediction with Artificial Neural Networks
# Using Sequential Neural Network Architecture
# Dündar Emre Özbirecikli
# 
#-------------------------------------------------------


import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler


def PreprocessData(data):
    """
    Preprocessing the data by encoding the labels and separating the train and test parts
    return: data sets for train and test
    """
    one_hot = OneHotEncoder()
    features = ['InitialFace', 'DropMethod', 'DropSurface']
    encoded_data = one_hot.fit_transform(data[features])
    encoded_data = encoded_data.toarray()
    encoded_data_columns= one_hot.get_feature_names_out(features)
    encoded_dataframe = pd.DataFrame(encoded_data, columns=encoded_data_columns)
    
    data = data.join(encoded_dataframe)
    data.drop(features, axis=1, inplace=True)
    
    label = LabelEncoder()
    y = label.fit_transform(data['FinalFace'])
    x = data.drop(['DropNumber', 'FinalFace', 'DropHeight'], axis=1)

    scale = StandardScaler()
    #x['DropAngle'] = scaler.fit_transform(x['DropAngle'])
    #x['BounceNum'] = scaler.fit_transform(x['BounceNum'])
    x[['DropAngle', 'BounceNum']] = scale.fit_transform(x[['DropAngle', 'BounceNum']])
    x.columns = x.columns.astype(str)
    
    oversampler = RandomOverSampler(random_state=4)
    x, y = oversampler.fit_resample(x, y)

    train_x , test_x, train_y, test_y = train_test_split(x, y, test_size=0.27, stratify=y, random_state=4)
    
    return train_x , test_x, train_y, test_y




def TrainTest(train_x, test_x, train_y, test_y):
    """
    Trains the data with the specified classifier then test it with the predicted output
    return: Accuracy score
    """
    def rate_schedule(epoch, lr):
        if epoch < 20:
            return lr
        else:
            return lr * np.exp(-1)
    
    lr_sch = LearningRateScheduler(rate_schedule)
    model = Sequential([Dense(64, activation='relu', input_shape=(train_x.shape[1],)), Dense(32, activation='relu'),
                        Dense(1, activation='sigmoid')])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=32, epochs=50,verbose=1,callbacks=[lr_sch])

    _, accuracy = model.evaluate(test_x, test_y, verbose=0)
    prediction = model.predict(test_x)
    pred_class = (prediction > 0.5).astype(int)

    return accuracy, pred_class




if __name__ == "__main__":
    # Reading the data from excel
    data = pd.read_excel("CoinDataset.xlsx")
    # Preprocessing the data for NEural Network
    train_x, test_x, train_y, test_y = PreprocessData(data)
    train_x.columns = train_x.columns.astype(str)
    test_x.columns = test_x.columns.astype(str)
    # Testing and Training
    accuracy, pred_class = TrainTest(train_x, test_x, train_y, test_y)

    # Printing the results
    conf_matrix = confusion_matrix(test_y, pred_class)
    class_report = classification_report(test_y, pred_class, target_names=['Head', 'Tail'])
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    print(f"Accuracy: {accuracy:.4f}")
        