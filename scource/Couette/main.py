import tensorflow as tf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

np.random.seed(1337)
tf.set_random_seed(1337)

delta = 1.
Nx = 40
Nc = 6
cw6 = pd.read_table('cw6.dat', sep=' ', header=None, engine='python')
cw6 = np.array(cw6)
c = cw6[:, 0:1]
c0 = c
c1 = -c
w = cw6[:, 1:2]

class PINN_Couette:
    def __init__(self, x_range, layers, learning_rate):
        self.x_range = x_range
        self.x_data_0 = np.linspace(x_range.min(), x_range.max(), 2*Nx+1)[:, None]
        self.x_data_1 = -self.x_data_0
        self.x_b = np.array([[x_range.min()],[x_range.max()]])

        self.layers = layers
        self.learning_rate = learning_rate
        self.step = 0
        # net initialization
        self.weights, self.biases = self.initialize_NN(layers)
        # definition of the placeholders
        self.xb_tf = tf.placeholder(tf.float32, shape=[None, self.x_b.shape[1]])
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x_data_0.shape[1]])
        self.x1_tf = tf.placeholder(tf.float32, shape=[None, self.x_data_0.shape[1]])
        # nn
        self.Phi0_pre, self.Phi0_x_pre = self.net_Phi(self.x0_tf)
        self.Phi1_pre, self.Phi1_x_pre = self.net_Phi(self.x1_tf)
        self.Phi_bc_pre, _ = self.net_Phi(self.xb_tf)
        # loss
        loss_1, loss_4 = self.loss_pde(self.Phi0_pre, self.Phi0_x_pre, self.Phi1_pre, self.Phi1_x_pre)
        loss_3 = (self.Phi_bc_pre[0, :Nc]-0.5)**2
        # loss_4 = (tf.matmul(self.Phi_bc_pre, w.astype(np.float32)) + 0.25)**2
        self.loss = tf.reduce_mean(loss_1)*30+ \
                    tf.reduce_mean(loss_3)+\
                    tf.reduce_mean(loss_4)

        # Optimizer
        self.optimizer1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer1.minimize(self.loss)
        self.optimizer2 = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                 method = "L-BFGS-B",
                                                                 options = {'maxiter': 50000,
                                                                            'ftol': 1.0*np.finfo(float).eps})

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.x_range.min()) / (self.x_range.max() - self.x_range.min()) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_Phi(self, x):
        Phi = self.neural_net(x, self.weights, self.biases)
        Phi_x = Phi[:, 0:1]
        for i in range(2*Nc):
            Phi_x = tf.concat([Phi_x, tf.gradients(Phi[:, i], x)[0]], 1)
        return Phi, Phi_x[:, 1:]

    def loss_pde(self, Phi0, Phi0_x, Phi1, Phi1_x):
        un0 = tf.matmul(Phi0[:, :Nc],w.astype(np.float32))- \
              tf.matmul(Phi1[:, :Nc], w.astype(np.float32))
        un1 = tf.matmul(Phi0, np.vstack((w.astype(np.float32), w.astype(np.float32))))
        # un1 = -un0
        loss_1 = Phi0[:, 0:1]*0
        for i in range(Nc):
            loss_1 = loss_1+(c[i] * Phi0_x[:, i:i+1] - delta * (un1 - Phi0[:, i:i+1])) ** 2
        loss_p = (Phi0[:, :Nc]+Phi1[:, Nc:])**2
        return loss_1/Nc, loss_p

    def callback(self, loss_):
        self.step += 1
        if self.step%100 == 0:
            print('Loss: %.3e'%(loss_))

    def train(self):
        tf_dict = {self.x0_tf: self.x_data_0,
                   self.x1_tf: self.x_data_1,
                   self.xb_tf: self.x_b}

        start_time = time.time()
        loss_value = 1
        it = 0
        while loss_value > 1e-3 and it < 2e4:
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.4e, Time: %.2f' % (it, loss_value, elapsed))
                start_time = time.time()
            it = it + 1

        self.optimizer2.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss], loss_callback=self.callback)
        #tf.train.Saver.save(self.sess, save_path='./Net/net.ckpt')

    def predict(self, x_star):
        tf_dict = {self.x0_tf: x_star,
                   self.x1_tf: -x_star}
        Phi0_pre = self.sess.run(self.Phi0_pre, tf_dict)
        Phi1_pre = self.sess.run(self.Phi1_pre, tf_dict)
        u0_pre = np.matmul(Phi0_pre[:, :Nc], w.astype(np.float32))- \
                 np.matmul(Phi1_pre[:, :Nc], w.astype(np.float32))
        return u0_pre, Phi0_pre

if __name__ == "__main__":
    x_range = np.array((-0.5, 0.5))
    model = PINN_Couette(x_range, layers=[1]+4*[40]+[2*Nc], learning_rate=1e-3)
    model.train()

    data = pd.read_table('./Couette/u_delta=1.0.txt', sep=' ', header=None, engine='python')
    u_dvm = np.array(data)
    x_star = np.linspace(-0.5, 0.5, 81)[:, None]
    u0, phi0 = model.predict(x_star)
    # np.savetxt('phi0.txt', phi0, fmt='%e')
    fig, ax = plt.subplots()
    ax.plot(x_star, u_dvm, 'k', linewidth=2.0)
    ax.plot(x_star, u0, 'r--', linewidth=2.0)
    plt.show()