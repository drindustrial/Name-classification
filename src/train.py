import numpy as np
import pandas as pd
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers

path = os.path.abspath(os.getcwd())
path = path[:path.rfind('\\') + 1] + "data\\data1\\"

eng_train = pd.read_csv(path + "train_eng.csv")
eng_train_sorted = eng_train.sort_values("Name")
clean_train = eng_train_sorted.copy()
prev = None
for index, row in eng_train_sorted.iterrows():
    if str(type(prev)) == "<class 'NoneType'>":
        prev = row
        continue
    if row["Name"] == prev["Name"]:
        clean_train = clean_train.drop(clean_train[clean_train['Name'] == row["Name"]].index)
    prev = row
clean_train = clean_train.sort_index()


unique = list(set("".join(clean_train["Name"])))
unique.sort()
vocab = dict(zip(unique, range(1,len(unique)+1)))

MAX_LEN = 15
def preproc(seq, voc, max_len = MAX_LEN):
    res = np.zeros(max_len)
    for i, ch in enumerate(seq):
        res[i] = voc[ch]
    return res

x_train = np.array([preproc(seq, vocab) for seq in clean_train["Name"]], dtype = np.int8)
y_train = np.array([0 if g == 'F' else 1 for g in clean_train["Gender"]], dtype = np.int8)


embedding_dim = 5

model = Sequential()
model.add(layers.Embedding(input_dim=len(vocab) + 1, 
                           output_dim=embedding_dim, 
                           input_length=MAX_LEN))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(15, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=3e-4, l2=3e-3),
    bias_regularizer=regularizers.l2(3e-3)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
print("Model #1 Dense layers")
print("Training...")
history = model.fit(x_train, y_train,
                    epochs=80,
                    verbose=0,
                    batch_size=128)
model.save("model1.h5")

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)

print(f"Training Accuracy:  {accuracy:.4f}")

model = Sequential()
model.add(layers.Embedding(input_dim=len(vocab) + 1, 
                           output_dim=embedding_dim, 
                           input_length=MAX_LEN))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=3e-4, l2=3e-3),
    bias_regularizer=regularizers.l2(3e-3)))
model.add(layers.Dropout(0.15))
model.add(layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=3e-4, l2=3e-3),
    bias_regularizer=regularizers.l2(3e-3)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

print("Model #2 MaxPool and Dense layers")
print("Training...")

history = model.fit(x_train, y_train,
                    epochs=50,
                    verbose=2,
                    batch_size=128)

model.save("model2.h5")
loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print(f"Training Accuracy:  {accuracy:.4f}")


model = Sequential()
model.add(layers.Embedding(input_dim=len(vocab) + 1, 
                           output_dim=embedding_dim, 
                           input_length=MAX_LEN))
model.add(layers.LSTM(14, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=2e-4, l2=2e-4),
    bias_regularizer=regularizers.l2(2e-4)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=2e-4, l2=2e-4),
    bias_regularizer=regularizers.l2(2e-4)))
model.add(layers.Dropout(0.15))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate = 0.008),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

print("Model #3 GRU and Dense layers")
print("Training...")

history = model.fit(x_train, y_train,
                    epochs=85,
                    verbose=0,
                    batch_size=128)

model.save("model3.h5")
loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print(f"Training Accuracy:  {accuracy:.4f}")