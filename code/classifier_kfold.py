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
from sklearn.model_selection import StratifiedKFold

os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"
#----------- Training -----------#
#Load training data and create train/val/test sets
fname_train = "/Users/prasunray/repos/M4R_JAN_OSKAR_PANEK/Labeled/full_data_covid_misinfo.csv"
test_split = 0.2
val_split = 0.2
df_train = pd.read_csv(fname_train)
df_test = df_train.sample(frac=test_split)
df_train.drop(df_test.index,inplace=True)
df_test = df_test[['cleaned_text','is_misinfo']]
df_train = df_train[['cleaned_text','is_misinfo']]
df_train.to_csv('train_bert.csv')
df_test.to_csv('test_bert.csv')


#Instantiate classifier
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # loss function used
metrics = tf.metrics.BinaryAccuracy()  # accuracy metrics

def build_classifier_model(loss,metrics,init_lr=0.001):
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')  # input
  preprocessing_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', name='preprocessing')  # preprocessing
  encoder_inputs = preprocessing_layer(text_input)  # preprocessed inputs to the encoder
  encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1', trainable=True, name='BERT_encoder')  # encoder
  outputs = encoder(encoder_inputs)  # run through BERT model
  net = outputs['pooled_output'] # output of BERT encoder
  net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)  # classification layer
  model = tf.keras.Model(text_input, net)
  optimizer = tf.keras.optimizers.RMSprop(init_lr)
  model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics)
  return model


#Train classifier
tf.random.set_seed(12345)

epochs = 3

batch_size = 16
init_lr = 0.00025

AUTOTUNE = tf.data.AUTOTUNE
# training and validation dataset

X_train = np.squeeze(df_train['cleaned_text'].to_numpy())
y_train = np.squeeze(df_train['is_misinfo'].to_numpy())
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_indices, val_indices) in enumerate(kfold.split(X_train, y_train)):
    print(f"Fold {fold + 1}")
    
    # Get training and validation data for this fold
    X_train_fold, X_val_fold = X_train[train_indices], X_train[val_indices]
    y_train_fold, y_val_fold = y_train[train_indices], y_train[val_indices]
    
    # Create and compile the model
    model = build_classifier_model(loss,metrics,init_lr)
    
    # Train the model
    model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    fold_metrics.append((val_loss, val_accuracy))

# Calculate and print average metrics across folds
avg_val_loss = np.mean([metrics[0] for metrics in fold_metrics])
avg_val_accuracy = np.mean([metrics[1] for metrics in fold_metrics])

print("Average Validation Loss:", avg_val_loss)
print("Average Validation Accuracy:", avg_val_accuracy)

if 1==1:
  final_classifier = model
  #Test results
  df_test = pd.read_csv('test_bert.csv')

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
  df = pickle.load(open('allTweetsDF.p','rb'))
  x = df['cleaned_text'].to_numpy()
  # Apply classifier
  predictions = np.squeeze(final_classifier.predict(x))
  pred_labels = predictions.copy()
  pred_labels[pred_labels>threshold]=1
  pred_labels[pred_labels<=threshold]=0
  df['misinfo_score']=predictions
  df['misinfo_class']=pred_labels
  # Update dataframe  