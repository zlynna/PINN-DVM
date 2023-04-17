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

class PINN:
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
        self.loss_1, self.loss_2 = self.loss_pde(self.Phi0_pre, self.Phi0_x_pre, self.Phi1_pre, self.Phi1_x_pre)
        self.loss_3 = self.loss_bc(self.Phi_bc_pre)
        self.loss_4 = tf.reduce_mean(tf.square(self.Phi_bc_pre[0:1, 2*Nc:3*Nc]-0.5))
        # loss_4 = (tf.matmul(self.Phi_bc_pre, w.astype(np.float32)) + 0.25)**2
        self.loss = tf.reduce_mean(self.loss_1)+\
                    tf.reduce_mean(self.loss_2)+\
                    tf.reduce_mean(self.loss_3)+\
                    tf.reduce_mean(self.loss_4)*100

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
        for i in range(4*Nc):
            Phi_x = tf.concat([Phi_x, tf.gradients(Phi[:, i], x)[0]], 1)
        return Phi, Phi_x[:, 1:]

    def loss_pde(self, Phi0, Phi0_x, Phi1, Phi1_x):
        w1 = w.astype(np.float32)
        r = tf.matmul(Phi0[:, :Nc], w1)-\
            tf.matmul(Phi1[:, :Nc], w1)
        t = 2/3*tf.matmul(Phi0[:, :Nc]*(c.T**2-0.5)+Phi0[:, 2*Nc:3*Nc], w1)-\
            2/3*tf.matmul(Phi1[:, :Nc]*(c.T**2-0.5)+Phi1[:, 2*Nc:3*Nc], w1)
        q = tf.matmul((Phi0[:, :Nc]*c.T**2+Phi0[:, 2*Nc:3*Nc])*c.T, w1)+\
            tf.matmul((Phi1[:, :Nc]*c.T**2+Phi1[:, 2*Nc:3*Nc])*c.T, w1)

        # un1 = -un0
        loss1 = Phi0[:, 0:1]*0
        loss2 = Phi0[:, 0:1]*0
        for i in range(Nc):
            H0 = r + (c[i]**2-0.5)*t+4/15*(c[i]**2-1.5)*c[i]*q
            H1 = t+4/15*c[i]*q
            loss1 = loss1+(c[i]*Phi0_x[:, i:i + 1]-delta*(H0-Phi0[:, i:i + 1]))**2
            loss2 = loss2+(c[i]*Phi0_x[:, i+2*Nc:i + 1+2*Nc]-delta*(H1-Phi0[:, i+2*Nc:i + 1+2*Nc]))**2
        loss_p = (Phi0[:, :Nc]+Phi1[:, Nc:2*Nc])**2+(Phi0[:, 2*Nc:3*Nc]+Phi1[:, 3*Nc:])**2
        return (loss1+loss2)/Nc, loss_p
    
    def loss_bc(self, Phi_bc):
        rhr = -2*np.sqrt(np.pi)*tf.matmul(tf.multiply(Phi_bc[1:2, :Nc], c.T.astype(np.float32)), w.astype(np.float32))
        loss5 = tf.square(Phi_bc[0:1, :Nc]-(rhr+0.5*(c.T**2-1)))
        return loss5

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
        while loss_value > 1e-3 and it < 1e4:
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_1 = self.sess.run(self.loss_1, tf_dict)
                loss_2 = self.sess.run(self.loss_2, tf_dict)
                loss_3 = self.sess.run(self.loss_3, tf_dict)
                loss_4 = self.sess.run(self.loss_4, tf_dict)
                print('It: %d, Loss: %.4e,Loss1: %.4e, Time: %.2f' % (it, loss_value,np.mean(loss_4), elapsed))
                start_time = time.time()
            it = it + 1

        self.optimizer2.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss], loss_callback=self.callback)
        #tf.train.Saver.save(self.sess, save_path='./Net/net.ckpt')

    def predict(self, x_star):
        tf_dict = {self.x0_tf: x_star,
                   self.x1_tf: -x_star}
        Phi0_pre = self.sess.run(self.Phi0_pre, tf_dict)
        Phi1_pre = self.sess.run(self.Phi1_pre, tf_dict)
        t0_pre = 2/3*np.matmul(Phi0_pre[:, :Nc]*(c.T**2-0.5)+Phi0_pre[:, 2*Nc:3*Nc], w)-\
                 2/3*np.matmul(Phi1_pre[:, :Nc]*(c.T**2-0.5)+Phi1_pre[:, 2*Nc:3*Nc], w)

        return t0_pre, Phi0_pre

if __name__ == "__main__":
    x_range = np.array((-0.5, 0.5))
    model = PINN(x_range, layers=[1]+4*[40]+[4*Nc], learning_rate=1e-3)
    model.train()

    data = pd.read_table('Fourier/t_delta=1.0.txt', sep=' ', header=None, engine='python')
    t_dvm = np.array(data)
    x_star = np.linspace(-0.5, 0.5, 81)[:, None]
    t0, phi0 = model.predict(x_star)
    # np.savetxt('phi0.txt', phi0, fmt='%e')
    fig, ax = plt.subplots()
    ax.plot(x_star, t_dvm, 'k', linewidth=2.0)
    ax.plot(x_star, t0, 'r--', linewidth=2.0)
    plt.show()