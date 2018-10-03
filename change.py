import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("data/", one_hot=True)
import pickle
sess = tf.InteractiveSession()
features = pickle.load(open("features.pkl", 'rb'))
labels = pickle.load(open("labels.pkl", 'rb'))

# Parameters
learning_rate = 0.01

#training_epochs = 1000
#batch_size = 30

training_epochs = 300
batch_size = 100

display_step = 2

# -----------------------------------------
#
# Two hidden layers
#
#------------------------------------------
'''
hidden 的个数可以进行调整

'''
# number of neurons in layer 1
n_hidden_1 = 250
# number of neurons in layer 2
n_hidden_2 = 200


#mnist data image of shape 28*28=784
input_size = 1506

# 0-9 digits recognition
output_size = 36

'''
可以改为layer1 layer2 进行增加layer
'''
def layer(x, weight_shape, bias_shape):
    
    # following the study by He et al. for ReLU layers
    w_std = (2.0/weight_shape[0])**0.5
    #w_std = 0.5  效果比上面差
    #print(weight_shape[0])
    w_0 = tf.random_normal_initializer(stddev=w_std)
	
    b_0 = tf.constant_initializer(value=0)
    
    W = tf.get_variable("W", weight_shape, initializer=w_0)
    b = tf.get_variable("b", bias_shape,   initializer=b_0)
    
    return tf.nn.relu(tf.matmul(x, W) + b)

def inference(x):
    
    with tf.variable_scope("hidden_layer_1"):
        hidden_1 = layer(x, [input_size, n_hidden_1], [n_hidden_1])
        #print([input_size, n_hidden_1])
     
    with tf.variable_scope("hidden_layer_2"):
        hidden_2 = layer(hidden_1, [n_hidden_1, n_hidden_2], [n_hidden_2])
        #print([n_hidden_1, n_hidden_2])
     
    with tf.variable_scope("output"):
        output = layer(hidden_2, [n_hidden_2, output_size], [output_size])
        #print([n_hidden_2, output_size])

    return output

def loss_1(output, y):
    # compute the average error per data sample 
    # by computing the cross-entropy loss over a minibatch
	
    dot_product = y * tf.log(output)
    #tf.reduce_sum: Computes the sum of elements across dimensions of a tensor.
    xentropy = -tf.reduce_sum(dot_product, reduction_indices=1)
    #tf.reduce_mean: Computes the mean of elements across dimensions of a tensor.
    loss = tf.reduce_mean(xentropy)

    return loss

def loss_2(output, y):
    
    #Computes softmax cross entropy between logits and labels.
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    loss = tf.reduce_mean(xentropy)

    return loss

def training(cost, global_step):

    tf.summary.scalar("cost", cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)

    return train_op


def evaluate(output, y):

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("validation error", (1.0 - accuracy))

    return accuracy

if __name__ == '__main__':
    
    #start_time = time.time()
    #log_files_path = 'C:/Users/Ali/logs/'

    #with tf.Graph().as_default():

        # image vector & label
        x = tf.placeholder("float", [None, input_size])   # mnist data image of shape 28*28=784
        y = tf.placeholder("float", [None, output_size])  # 0-9 digits recognition

        output = inference(x)

        cost = loss_2(output, y)
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = training(cost, global_step)
        
        #train_op = training(cost, global_step=None)

        eval_op = evaluate(output, y)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        sess = tf.Session()
        
        #https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter
        summary_writer = tf.summary.FileWriter(log_files_path, sess.graph)
        
        
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Training cycle
        for epoch in range(training_epochs):

            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            
            # Loop over all batches
            for i in range(total_batch):
			
			
                minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                # Fit training using batch data
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y})/total_batch
                
            # Display logs per epoch step
            if epoch % display_step == 0:

                accuracy = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

                print("Epoch:", '%03d' % (epoch+1), "cost function=", "{:0.7f}".format(avg_cost), " Validation Error:", (1.0 - accuracy))
                
                summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
                summary_writer.add_summary(summary_str, sess.run(global_step))
                
                #https://www.tensorflow.org/api_docs/python/tf/train/Saver
                saver.save(sess, log_files_path+'model-checkpoint', global_step=global_step)


        print("Optimization Finished!")
        accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Test Accuracy:", accuracy)
        
        elapsed_time = time.time() - start_time
        print('Execution time was %.3f' % elapsed_time)