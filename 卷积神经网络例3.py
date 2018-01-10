from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义值
x_train_images = mnist_data.train.images
y_train_labels = mnist_data.train.labels

# 定义batch_size
my_batch_size = 100

# 定义batch_num
num_batch = len(mnist_data.train.images) // my_batch_size
print(num_batch)

x = tf.placeholder(tf.float32, [None, 784], name='x-input')
y = tf.placeholder(tf.float32, [None, 10], name='y-input')
x_images = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一层
w_cov1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_cov1 = tf.Variable(tf.zeros([32]))
h_vov1 = tf.nn.conv2d(x_images, w_cov1, strides=[1, 1, 1, 1], padding='SAME')
h_act1 = tf.nn.relu(h_vov1 + b_cov1)
h_pool1 = tf.nn.max_pool(h_act1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 定义第二层
w_cov2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_cov2 = tf.Variable(tf.zeros([64]))
h_vov2 = tf.nn.conv2d(h_pool1, w_cov2, strides=[1, 1, 1, 1], padding='SAME')
h_act2 = tf.nn.relu(h_vov2 + b_cov2)
h_pool2 = tf.nn.max_pool(h_act2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 全连接层
w_full1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_full1 = tf.Variable(tf.zeros([1024]))

# 扁平化
h_pool2_flatten = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# 第一个全连接层输出
flatten_out = tf.nn.relu(tf.matmul(h_pool2_flatten, w_full1) + b_full1)

# 抓爆
keep_prob = tf.placeholder(tf.float32)
flatten_dropout = tf.nn.dropout(flatten_out, keep_prob)

# 第二个全连接层
w_full2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_full2 = tf.Variable(tf.zeros([10]))

predictions = tf.nn.softmax(tf.matmul(flatten_dropout, w_full2) + b_full2)

# 定义损失
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions))  # 定义准确度
# 训练步骤
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_loss)

correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(predictions, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20):
        for batch in range(num_batch):
            x_train_images_batch, y_train_labels_batch = mnist_data.train.next_batch(my_batch_size)
            sess.run(train_step, feed_dict={x: x_train_images_batch, y: y_train_labels_batch, keep_prob: 0.7})
        list1 = []
        for i in range(10):

            test_data=mnist_data.test.next_batch(1000)
            acc_test = sess.run(accuracy, feed_dict={x: test_data[0],
                                                 y: test_data[1], keep_prob: 1.0})
            list1.append(acc_test)
        acc_test_mean=sum(list1)/10
        print('Iter:' + str(step) + 'test_acc:' + str(acc_test_mean))

# 写在后面：这个数据集最多是1000*28*28*28
# 再大内存不够
# 出错的地方会是在conv2D
# 显示为55000*28*28*64
