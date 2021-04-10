import numpy as np
import pandas as pd
import tensorflow as tf

eng_train = pd.read_csv("train_eng.csv")
eng_test = pd.read_csv("test_eng.csv")
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
x_test = np.array([preproc(seq, vocab) for seq in eng_test["Name"]], dtype = np.int8)
y_train = np.array([0 if g == 'F' else 1 for g in eng_train["Gender"]], dtype = np.int8)
y_test = np.array([0 if g == 'F' else 1 for g in eng_test["Gender"]], dtype = np.int8)

for model_name in ["model1", "model2", "model3"]:
    model = tf.keras.models.load_model(model_name + '.h5')
    print(model_name)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print(f"Testing Accuracy:  {accuracy:.4f} \n\n")