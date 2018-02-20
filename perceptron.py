+# -*- coding: utf-8 -*- 


+""" 


+Created on Sun Feb 18 19:58:34 2018 


+ 


+@author: qoliver 


+""" 


+ 


+from __future__ import print_function 


+ 


+# Import MNIST data 


+#from tensorflow.examples.tutorials.mnist import input_data 


+#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) 


+ 


+import tensorflow as tf 


+ 


+# Parameters 


+learning_rate = 0.001 


+training_epochs = 1 


+batch_size = 10 


+display_step = 1 


+ 


+# Network Parameters 


+n_hidden_1 = 256 # 1st layer number of neurons 


+#n_hidden_2 = 256 # 2nd layer number of neurons 


+n_input = 9 # MNIST data input (img shape: 28*28) 


+n_classes = 2 # MNIST total classes (0-9 digits) 


+ 


+# tf Graph input 


+X = tf.placeholder("float", [None, n_input]) 


+Y = tf.placeholder("float", [None, n_classes]) 


+ 


+# Store layers weight & bias 


+weights = { 


+    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])), 


+#    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])), 


+    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes])) 


+} 


+biases = { 


+    'b1': tf.Variable(tf.random_normal([n_hidden_1])), 


+ #   'b2': tf.Variable(tf.random_normal([n_hidden_2])), 


+    'out': tf.Variable(tf.random_normal([n_classes])) 


+} 


+ 


+ 


+# Create model 


+def multilayer_perceptron(x): 


+    # Hidden fully connected layer with 256 neurons 


+    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1']) 


+    # Hidden fully connected layer with 256 neurons 


+   # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']) 


+    # Output fully connected layer with a neuron for each class 


+    out_layer = tf.matmul(layer_1, weights['out']) + biases['out'] 


+    return out_layer 


+ 


+# Construct model 


+logits = multilayer_perceptron(X) 


+training_data=[] 


+training_labels=[] 


+for i in range(100): 


+    t=trainingSetMakerTTT() 


+    training_data.append(t[0][0]) 


+    training_labels.append(t[0][1]) 


+train_data=np.array(training_data) 


+train_labels=[] 


+for label in training_labels: 


+    t=[label] 


+    train_labels.append(t) 


+train_labels=np.array(train_labels) 


+ 


+testing_data=[] 


+testing_labels=[] 


+for i in range(100): 


+    t=trainingSetMakerTTT() 


+    testing_data.append(t[0][0]) 


+    testing_labels.append(t[0][1]) 


+eval_data=np.array(training_data) 


+eval_labels=np.array([training_labels]) 


+ 


+ 


+# Define loss and optimizer 


+loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( 


+    logits=logits, labels=Y)) 


+optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 


+train_op = optimizer.minimize(loss_op) 


+# Initializing the variables 


+init = tf.global_variables_initializer() 


+ 


+with tf.Session() as sess: 


+    sess.run(init) 


+ 


+    # Training cycle 


+    for epoch in range(training_epochs): 


+        avg_cost = 0. 


+        total_batch = int(len(train_data)/batch_size) 


+        # Loop over all batches 


+        for i in range(total_batch): 


+            #todo: set x to input y to label in batchs indexed by batch_size*i,batch_size*i+100? 


+            batch_x =  


+            batch_y = train_data.train.next_batch(batch_size) 


+            # Run optimization op (backprop) and cost op (to get loss value) 


+            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, 


+                                                            Y: batch_y}) 


+            # Compute average loss 


+            avg_cost += c / total_batch 


+        # Display logs per epoch step 


+        if epoch % display_step == 0: 


+            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost)) 


+    print("Optimization Finished!") 


+ 


+    # Test model 


+    pred = tf.nn.softmax(logits)  # Apply softmax to logits 


+    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1)) 


+    # Calculate accuracy 


+    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 


+ 


+#todo fixe print out put on separate eval set of tic tac toe data 


+    print("Accuracy:", accuracy.eval({X: #some tic tac input, 


+                                      Y: #some tic tac labels 


+                                      })) 


+ 


+ 