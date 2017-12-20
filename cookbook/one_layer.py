'''
真是个巨大的问题！！书上的代码训练不收敛~~我还没找到原因ing

'''



import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# sklearn自带一些数据集：学会怎么拿过来用
from sklearn import datasets
iris = datasets.load_iris() # 加载鸢尾花数据集
# digits = datasets.load_digits()

x_vals = np.array([x[0:3] for x in iris.data]) # 150x3
y_vals = np.array([x[3] for x in iris.data]) # 150x1

sess = tf.Session()

seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

# train-test: 80%-20%
train_indices = np.random.choice(len(x_vals),round(len(x_vals) * 0.8), replace=False) # 不放回抽样
test_indices = np.array(list( set(range(len(x_vals))) - set(train_indices)) ) # 使用set去重

x_vals_train = x_vals[train_indices] # 索引数组，获得train_indices指示的下标的值
x_vals_test = x_vals[test_indices]

y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# 上面是准备数据

# 标准化数据
def normalize_cols(m):
	col_max = m.max(axis=0) # 拿到每列最大的值
	col_min = m.min(axis=0) # 拿到每列最小的值

	res = (m - col_min) / (col_max - col_min) # 将数据标准化到每列最大最小之间
	return res

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train)) # 标准化 + nan变0
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# batch-size
batch_size = 50 
x_data = tf.placeholder(shape=[None,3], dtype=tf.float32) # 行数不定，具有可扩展性
y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)

# 进一步定义网络参数
hidden_layer_nodes = 5 # 隐层结点数量
# 权重的定义和Ng的课上说的有些出入，是因为输入的数据不是按照列向量堆叠（每一列是个输入），而是每一行是一个输入
# 
A1 = tf.Variable(tf.random_normal(shape=[3,hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))

A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1]))
b2 = tf.Variable(tf.random_normal(shape=[1]))

# declare model
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data,A1),b1)) # 注意如果按照Ng的输入方法，tf.matmul(w.T,X),这里是反过来的，掌握原理很重要
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output,A2),b2))

# loss 函数
loss = tf.reduce_mean(tf.square(y_target - final_output))

# optimizer
my_opt = tf.train.AdagradOptimizer(0.005) # 学习率
train_step = my_opt.minimize(loss) # 优化损失函数

init = tf.global_variables_initializer() # 初始化
print('initialize_all_variables')
sess.run(init) # 进行初始化

loss_vec = []
test_loss = []

for i in range(500):

	rand_index = np.random.choice(len(x_vals_train),size=batch_size) #每次随机选择batch_size多个样本进行训练
	rand_x = x_vals_train[rand_index]
	rand_y = np.transpose([y_vals_train[rand_index]]) # ground truth, 转置成列向量

	sess.run(train_step, feed_dict={ x_data:rand_x, y_target:rand_y})

	# 保存训练误差
	temp_loss = sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})
	loss_vec.append(np.sqrt(temp_loss))

	# 保存测试误差
	test_temp_loss = sess.run(loss, feed_dict={x_data:x_vals_test, y_target:np.transpose([y_vals_test])})
	test_loss.append(np.sqrt(test_temp_loss))

	if (i + 1) % 50 == 0: # 每50次输出一次
		print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))

# plot
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss,'r--',label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')

plt.legend(loc='upper right')
plt.show()

