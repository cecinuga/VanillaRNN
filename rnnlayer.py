class RNNLayer(tf.Module):
    def __init__(self, units, input_dim, output_dim):
        self.units = units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.builded = False   

    @tf.function
    def build(self):
        if not self.builded:
            self.W_hx = tf.Variable(tf.eye(self.units, self.input_dim), trainable=True, dtype=tf.float32)
            self.W_hh = tf.Variable(tf.eye(self.units, self.units), trainable=True, dtype=tf.float32)
            self.W_hy = tf.Variable(tf.eye(self.input_dim, self.units), trainable=True, dtype=tf.float32)
            self.h = tf.Variable(tf.zeros([self.units, 1]), dtype=tf.float32)
            self.vars = [self.W_hx, self.W_hh, self.W_hy]
            self.builded = True

    @tf.function(reduce_retracing=True)
    def __call__(self, x):
        print("------LSTMLayer------")
        print("[#] W_hx: {0}, W_hh: {1}, W_hy: {2}, X: {3}".format(self.W_hx.shape, self.W_hh.shape, self.W_hy.shape, x.shape))
        updated_input = tf.multiply(self.W_hx, x)
        updated_memory = tf.matmul(self.W_hh, self.h)
        print("[#] updated_input: {0}, updated_memory: {1}".format(updated_input.shape, updated_memory.shape))
        self.h = tf.math.tanh(
            tf.add(updated_memory, updated_input), 
        )
        output = tf.reduce_sum(tf.nn.sigmoid(tf.multiply(self.W_hy, tf.transpose(self.h))), axis=1)
        print("[#] h: {0}, output: {1}".format(self.h.shape, output.shape))
        print("\n")
        return output, self.h