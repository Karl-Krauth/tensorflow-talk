from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Import the dataset.
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Define placeholder variables.
inputs = tf.placeholder(tf.float32, [None, 28 * 28], name="inputs")
outputs = tf.placeholder(tf.float32, [None, 10], name="outputs")

# Model variables that will get optimized.
weights = tf.Variable(tf.zeros([10, 784]), name="weights")
biases = tf.Variable(tf.zeros([10, 1]), name="biases")

# Output variable.
weighted_sum = tf.matmul(weights, tf.transpose(inputs)) + biases
model_outputs = tf.transpose(tf.nn.softmax(weighted_sum), name="model_outputs")

# Cost function.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(outputs * tf.log(model_outputs), 1),
                               name="cross_entropy")

# Optimizer setup
optimizer = tf.train.GradientDescentOptimizer(0.5)
train_op = optimizer.minimize(cross_entropy)

# Tensorboard setup
tf.histogram_summary("weights", weights)
tf.histogram_summary("biases", biases)
tf.scalar_summary("cross_entropy", cross_entropy)
merge_op = tf.merge_all_summaries()

# Run the computational graph.
session = tf.Session()
writer = tf.train.SummaryWriter("summary_dir", graph=session.graph)
session.run(tf.initialize_all_variables())

for i in xrange(1000):
    # Get the next batch in the training data.
    input_batch, output_batch = mnist_data.train.next_batch(100)
    # Summarize the current state of the model.
    summary = session.run(merge_op, feed_dict={inputs: input_batch,
                                               outputs: output_batch})
    # Log summary info to disk.
    writer.add_summary(summary, i)
    # Run the optimizer for one step.
    session.run(train_op, feed_dict={inputs: input_batch,
                                     outputs: output_batch})
    if i % 10 == 0:
        print i

predictions = tf.equal(tf.argmax(model_outputs, 1), tf.argmax(outputs, 1), name="predictions")
accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32), name="accuracy")
print "Accuracy is: ", session.run(accuracy, feed_dict={inputs: mnist_data.test.images,
                                                        outputs: mnist_data.test.labels})
session.close()

