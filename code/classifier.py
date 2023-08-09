"""
Train and apply vaccine misinfo classifier
"""

#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle
import json
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


#----------- Training -----------#
#Load training data and create train/val/test sets
fname_train = "/Users/prasunray/repos/M4R_JAN_OSKAR_PANEK/Labeled/full_data_covid_misinfo.csv"
test_split = 0.2
val_split = 0.2
dftrain = pd.read_csv(fname_train)
dftest = dftrain.sample(frac=test_split)
dftrain.drop(dftest.index,inplace=True)
dftest = dftest[['cleaned_text','is_misinfo']]
dftrain = dftrain[['cleaned_text','is_misinfo']]
dftrain.to_csv('train_bert.csv')
dftest.to_csv('test_bert.csv')
dftest = None;dftrain= None

#Instantiate classifier
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # loss function used
metrics = tf.metrics.BinaryAccuracy()  # accuracy metrics

def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')  # input
  preprocessing_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', name='preprocessing')  # preprocessing
  encoder_inputs = preprocessing_layer(text_input)  # preprocessed inputs to the encoder
  encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1', trainable=True, name='BERT_encoder')  # encoder
  outputs = encoder(encoder_inputs)  # run through BERT model
  net = outputs['pooled_output'] # output of BERT encoder
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)  # classification layer
  return tf.keras.Model(text_input, net)


#Train classifier
tf.random.set_seed(12345)

epochs = 3

batch_size = 16

AUTOTUNE = tf.data.AUTOTUNE
# training and validation dataset
train_val_data = tf.data.experimental.CsvDataset(["train_bert.csv"], [tf.string,tf.int32], select_cols=[1,2],header=True)
train_val_data.shuffle(buffer_size=len(list(train_val_data)))
train_val_set_size = len(list(train_val_data))
val_n = int(val_split*train_val_set_size)

train_data = train_val_data.skip(val_n).batch(batch_size)
train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = train_val_data.take(val_n).batch(batch_size)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

steps_per_epoch = train_val_set_size - val_n
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

#optimizer = optimization.create_optimizer(init_lr=init_lr,
#                                          num_train_steps=num_train_steps,
#                                          num_warmup_steps=num_warmup_steps,
#                                          optimizer_type='adamw')

init_lr = 0.00025
#optimizer = tf.keras.optimizers.AdamW()
optimizer = tf.keras.optimizers.legacy.Adam()
optimizer = tf.keras.optimizers.RMSprop(init_lr)

final_classifier = build_classifier_model()
final_classifier.compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics)
history = final_classifier.fit(x=train_data, validation_data=val_data, epochs=epochs)

final_classifier.save_weights('final_classifier8')

#Test results
df_test = pd.read_csv('test_bert .csv')

# test_data = tf.data.experimental.CsvDataset(["test_bert.csv"], [tf.string,tf.int32] ,select_cols=[2,3])
# test_data = test_data.batch(batch_size)
# test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)
x = df_test['cleaned_text'].to_numpy()
y = df_test['is_misinfo'].to_numpy()
test_results = final_classifier.evaluate(x,y)
test_predictions = np.squeeze(final_classifier.predict(x))

#ROC curve
df_test['predictions'] = test_predictions
true_labels = df_test['is_misinfo']
predicted_labels = df_test['predictions']
fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)

roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='red', label='Random classifier line')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

threshold = thresholds[np.argmax(tpr - fpr)]  # optimal threshold

pred_labels = df_test['predictions'].apply(lambda x: 1 if x > threshold else 0)
print(confusion_matrix(true_labels, pred_labels))
print(accuracy_score(true_labels, pred_labels))
print(precision_score(true_labels, pred_labels))
print(recall_score(true_labels, pred_labels))
print(f1_score(true_labels, pred_labels))

#----------- Classification -----------#
# Load tweet dataframe


# Apply classifier

 
# Update dataframe  