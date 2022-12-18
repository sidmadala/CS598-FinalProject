# %%
# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import scipy
from IPython.display import Audio


# %%
# Import car noise data and label based on title

path = "../../data/paws/"
keyword = "car"

# Load all audio files into tuples of (data, car) where car is a boolean based on file title
# Remove sample rate from data
def load_data(path, keyword):
    data = []
    for file in os.listdir(path):
        if file.endswith(".wav"):
            if keyword in file:
                data.append((scipy.io.wavfile.read(path + file), True))
            else:
                data.append((scipy.io.wavfile.read(path + file), False))
    return data

data = load_data(path, keyword)


# Remove sample rate from data
data = [(np.array(x[0][1]), x[1]) for x in data]

# Make all data the same length
max_len = max([len(x[0]) for x in data])
data = [(np.pad(x[0], (0, max_len - len(x[0])), 'constant'), x[1]) for x in data]

# Scale data to be between 0 and 1
data = [(x[0] / np.max(x[0]), x[1]) for x in data]

# Convert to pandas dataframe
df = pd.DataFrame(data, columns=["data", "car"])

# Display example data
df.head()

# %%
# Split into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df["data"], df["car"], test_size=0.2, random_state=42)

# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(X_train)

# %%
# Train SVM model

from sklearn.svm import SVC

model = SVC(kernel="linear", class_weight="balanced")
model.fit(list(X_train), y_train)

# %%
# Test model
predictions = model.predict(list(X_test))

# Evaluate model
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
print(accuracy)

# %%
# Save model

import pickle

pickle.dump(model, open("model.pkl", "wb"))

# Load model

model = pickle.load(open("model.pkl", "rb"))

# %%
# Create confusion matrix for model

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)
print(cm)

# Display confusion matrix properly

import seaborn as sn

df_cm = pd.DataFrame(cm, index = [i for i in ["Not Car", "Car"]],
columns = [i for i in ["Not Car", "Car"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()