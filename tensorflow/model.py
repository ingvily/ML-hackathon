import tensorflow as tf

    
class FFNN(object):
	def __init__(self, arch, train=True, dtype=tf.float32):
		self.arch = arch
		self.train = train
		self.dtype = dtype

	def build(self):
		with tf.variable_scope('default'):
			self.build_inference()
			if self.train:
				self.build_loss()
				self.build_training()

	def build_inference(self):
		self.params = []

		def build_flat(name, size, prevl, a=tf.nn.relu):
			with tf.variable_scope(name) as scope:
				w = tf.get_variable(
					'w',
                    shape=[prevl.get_shape().as_list()[1], size],
                    initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=self.dtype),
                    dtype=self.dtype)
				self.params.append(w)
				b = tf.get_variable(
                    'b',
                    shape=[size],
                    initializer=tf.constant_initializer(0.1, dtype=self.dtype),
                    dtype=self.dtype)
				self.params.append(b)
				l = a(tf.matmul(prevl, w) + b, name=name)
				return l

		self.x = prevl = tf.placeholder(self.dtype, shape=[None, self.arch[0]], name='input')
		
		for (l, size) in enumerate(self.arch[1:-1]):
			prevl = build_flat('hidden{}'.format(l), size, prevl)
			print prevl
		self.logits = build_flat('logits', self.arch[-1], prevl, a=tf.identity)
		self.y = tf.nn.softmax(self.logits, name='output')


	def build_loss(self):
		self.ideal = tf.placeholder(self.dtype, shape=self.y.get_shape(), name='ideal')
		xent = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ideal, name='xent_per_example')
		self.xent_mean = tf.reduce_mean(xent, name='xent')

	def build_training(self):
		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.exponential_decay(0.0005, global_step, 500, 0.96, staircase=False)
		self.train = tf.train.RMSPropOptimizer(lr).minimize(self.xent_mean)