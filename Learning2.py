# -*- coding: utf-8 -*-
"""
https://www.tensorflow.org/tutorials/layers
First layer
MNIST
Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU ( Rectified Linear) activation function
TicTac
Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU ( Rectified Linear) activation function

Second layer Parameter goal make sure stride prevents overlap
MNIST
Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)
TiTac
Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)

Third Layer
MNIST
Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
TicTac

fourth layer
MNIST
Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
TicTac

fifth layer
MNIST
Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4 (probability of 0.4 that any given element will be dropped during training)
TicTac
Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4 (probability of 0.4 that any given element will be dropped during training)

sixth layer
MNIST
Dense Layer #2 (Logits Layer): 10 neurons, one for each digit target class (0–9).
TicTac
Dense Layer #2 (Logits Layer): 2 neurons, one for each digit target class (tie, not tie).

"""
import tensorflow as tf
import numpy as np
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 10, 10, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=8,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=0)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=8,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=0)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 3*3*8])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


  # Load training and eval data

training_data=[]
training_labels=[]
for i in range(100):
    t=trainingSetMakerTTT()
    training_data.append(np.float16(t[0][0][0][0]))
    training_labels.append(np.float16(t[0][0][0][1]))
train_data=np.array(training_data)
train_labels=[]
for label in training_labels:
    t=[label]
    train_labels.append(t)
train_labels=np.array(train_labels)

testing_data=[]
testing_labels=[]
for i in range(100):
    t=trainingSetMakerTTT()
    testing_data.append(np.float16(t[0][0][0][0]))
    testing_labels.append(np.float16(t[0][0][0][1]))
eval_data = np.array([training_data])
eval_labels=np.array([training_labels])

mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  
  
train_input_fn = tf.estimator.inputs.numpy_input_fn(
   x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])


eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)