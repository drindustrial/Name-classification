import numpy as np
import pandas as pd

eng_train = pd.read_csv("train_eng.csv")
unique = list(set("".join(eng_train["Name"])))
unique.sort()
vocab = dict(zip(unique, range(1,len(unique)+1)))

MAX_LEN = 15
def preproc(seq, voc, max_len = MAX_LEN):
    res = np.zeros(max_len)
    for i, ch in enumerate(seq):
        res[i] = voc[ch]
    return res

x_train = np.array([preproc(seq, vocab) for seq in eng_train["Name"]], dtype = np.int8)
y_train = np.array([0 if g == 'F' else 1 for g in eng_train["Gender"]], dtype = np.int8)

ffrom tensorflow.keras.models import Sequential
from tensorflow.keras import layers

embedding_dim = 9

model = Sequential()
model.add(layers.Embedding(input_dim=len(vocab) + 1, 
                           output_dim=embedding_dim, 
                           input_length=MAX_LEN))
model.add(layers.Flatten())
model.add(layers.Dense(15, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
print("Model #1 Dense layers")
print("Training...")
history = model.fit(x_train, y_train,
                    epochs=25,
                    verbose=2,
                    batch_size=128)
model.save("model1.h5")

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)

print(f"Training Accuracy:  {accuracy:.4f}")

model = Sequential()
model.add(layers.Embedding(input_dim=len(vocab) + 1, 
                           output_dim=embedding_dim, 
                           input_length=MAX_LEN))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(35, activation='relu'))
model.add(layers.Dense(35, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

print("Model #2 MaxPool and Dense layers")
print("Training...")

history = model.fit(x_train, y_train,
                    epochs=25,
                    verbose=2,
                    batch_size=128)

model.save("model2.h5")
loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print(f"Training Accuracy:  {accuracy:.4f}")




model = Sequential()
model.add(layers.Embedding(input_dim=len(vocab) + 1, 
                           output_dim=embedding_dim, 
                           input_length=MAX_LEN))
model.add(layers.LSTM(15, activation='relu'))
model.add(layers.Dropout(0.15))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

print("Model #3 GRU and Dense layers")
print("Training...")

history = model.fit(x_train, y_train,
                    epochs=25,
                    verbose=2,
                    batch_size=128)

model.save("model3.h5")
loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print(f"Training Accuracy:  {accuracy:.4f}")