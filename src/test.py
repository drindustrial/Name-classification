import numpy as np
import pandas as pd
import tensorflow as tf
import os

path = os.path.abspath(os.getcwd())
path = path[:path.rfind('\\') + 1] + "data\\data1\\"

eng_train = pd.read_csv(path + "train_eng.csv")
eng_test = pd.read_csv(path + "test_eng.csv")
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

eng_test_sorted = eng_test.sort_values("Name")
clean_test = eng_test_sorted.copy()
prev = None
for index, row in eng_test_sorted.iterrows():
    if str(type(prev)) == "<class 'NoneType'>":
        prev = row
        continue
    if row["Name"] == prev["Name"]:
        clean_test = clean_test.drop(clean_test[clean_test['Name'] == row["Name"]].index)
    prev = row
clean_test = clean_test.sort_index()

unique = list(set("".join(clean_train["Name"])))
unique.sort()
vocab = dict(zip(unique, range(1,len(unique)+1)))
MAX_LEN = 15
def preproc(seq, voc, max_len = MAX_LEN):
    res = np.zeros(max_len)
    for i, ch in enumerate(seq):
        res[i] = voc[ch]
    return res

x_test = np.array([preproc(seq, vocab) for seq in clean_test["Name"]], dtype = np.int8)
y_test = np.array([0 if g == 'F' else 1 for g in clean_test["Gender"]], dtype = np.int8)

for model_name in ["model1", "model2", "model3"]:
    model = tf.keras.models.load_model(model_name + '.h5')
    print(model_name)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print(f"Testing Accuracy:  {accuracy:.4f} \n\n")