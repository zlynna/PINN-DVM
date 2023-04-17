import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from hyper_parameters import global_var
from phi1 import phi1_cul

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

np.random.seed(42)
tf.random.set_seed(42)

### import data 
xx = global_var.get_value('xx')
yy = global_var.get_value('yy')
Nx = global_var.get_value('Nx')
Ny = global_var.get_value('Ny')
Nvx = global_var.get_value('Nvx')
Nvy = global_var.get_value('Nvy')
X_data = global_var.get_value('X')
mu = global_var.get_value('mu')
theta = global_var.get_value('theta')
d_mu = global_var.get_value('d_mu')
d_theta = global_var.get_value('d_theta')
X_t = global_var.get_value('X_t')
X_train = global_var.get_value('X_train')
delta = global_var.get_value('delta')
inp1 = global_var.get_value('inp1')
inp2 = global_var.get_value('inp2')
inp3 = global_var.get_value('inp3')
inp4 = global_var.get_value('inp4')
Xbt = global_var.get_value('X1')
Xbb = global_var.get_value('X2')
Xbl = global_var.get_value('X3')
Xbr = global_var.get_value('X4')
Nxb = global_var.get_value('Nxb')
mu = mu.astype(np.float32)
theta = theta.astype(np.float32)
d_mu = d_mu.astype(np.float32)
d_theta = d_theta.astype(np.float32)

layer_sizes=[2]+[80]*6+[Nvx*Nvy]

def neural_net(layer_sizes):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    for i, width in enumerate(layer_sizes[1:-1]):
        model.add(layers.Dense(
            width, activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/(layer_sizes[i+1] + layer_sizes[i+2])), seed=None),
            bias_initializer="zeros"))
    model.add(layers.Dense(
            layer_sizes[-1], activation=None,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/(layer_sizes[i+1] + layer_sizes[i+2])), seed=None),
            bias_initializer="zeros"))
    return model

@tf.function
def net_domain(x):   
    x = 2.0 * (x - X_data.min(axis=0)) / (X_data.max(axis=0) - X_data.min(axis=0)) - 1.0 
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        phi = Phi_model(x) # Nx*Ny, Nvx*Nvy
        psi = Psi_model(x)
    # grad for phi1
    phi_xy = g.batch_jacobian(phi, x)
    phi_x = tf.reshape(phi_xy[..., 0], shape=(-1, Nvy, Nvx))*2
    phi_y = tf.reshape(phi_xy[..., 1], shape=(-1, Nvy, Nvx))*2
    # grad for psi
    psi_xy = g.batch_jacobian(psi, x)
    psi_x = tf.reshape(psi_xy[..., 0], shape=(-1, Nvy, Nvx))*2
    psi_y = tf.reshape(psi_xy[..., 1], shape=(-1, Nvy, Nvx))*2
    return tf.reshape(phi, shape=(-1, Nvy, Nvx)), tf.reshape(psi, shape=(-1, Nvy, Nvx)), phi_x, phi_y, psi_x, psi_y

@tf.function
def net_Phi_bc(x):
    x = 2.0 * (x - X_data.min(axis=0)) / (X_data.max(axis=0) - X_data.min(axis=0)) - 1.0 
    phi = Phi_model(x) # Nx*Ny, Nvx*Nvy
    psi = Psi_model(x)
    return tf.reshape(phi, shape=(-1, Nvy, Nvx)), tf.reshape(psi, shape=(-1, Nvy, Nvx))

### loss ###
def loss(x_batch, xbl, xbr, xbt, xbb, phi1):
    Phi_pre, Psi_pre, Phi_x_pre, Phi_y_pre, Psi_x_pre, Psi_y_pre = net_domain(x_batch)
    Eq_phi, Eq_psi = Eq_comp(Phi_x_pre, Phi_y_pre, Psi_x_pre, Psi_y_pre, Psi_pre, Phi_pre, phi1)

    Phi_bct_pre, psi_bct_pre = net_Phi_bc(xbt)
    Phi_bcb_pre, psi_bcb_pre = net_Phi_bc(xbb)
    Phi_bcl_pre, psi_bcl_pre = net_Phi_bc(xbl)
    Phi_bcr_pre, psi_bcr_pre = net_Phi_bc(xbr)
    phi_t, phi_b, phi_l, phi_r, psi_t, psi_b, psi_l, psi_r = Bc(Phi_bct_pre, psi_bct_pre,
                                                                Phi_bcb_pre, psi_bcb_pre,
                                                                Phi_bcl_pre, psi_bcl_pre,
                                                                Phi_bcr_pre, psi_bcr_pre)
    mse_eq_phi = tf.reduce_mean(tf.square(Eq_phi))*phi_w
    mse_eq_psi = tf.reduce_mean(tf.square(Eq_psi))*psi_w
    mse_bc_t = tf.reduce_mean(tf.square(phi_t))*phi_t_w

    mse_bc_steady = tf.reduce_mean(tf.square(phi_b))*phi_b_w+\
                    tf.reduce_mean(tf.square(phi_l))*phi_b_w+\
                    tf.reduce_mean(tf.square(phi_r))*phi_b_w
    mse_bc_psi = tf.reduce_mean(tf.square(psi_t))*psi_t_w+\
                 tf.reduce_mean(tf.square(psi_b))*psi_b_w+\
                 tf.reduce_mean(tf.square(psi_l))*psi_l_w+\
                 tf.reduce_mean(tf.square(psi_r))*psi_r_w
    mse_sum = mse_eq_phi+mse_eq_psi+mse_bc_t+mse_bc_steady+mse_bc_psi

    mse_eq_phi = tf.reduce_mean(tf.square(Eq_phi))
    mse_eq_psi = tf.reduce_mean(tf.square(Eq_psi))
    mse_bc_t = tf.reduce_mean(tf.square(phi_t))
    mse_bc_steady = tf.reduce_mean(tf.square(phi_b))+\
                    tf.reduce_mean(tf.square(phi_l))+\
                    tf.reduce_mean(tf.square(phi_r))
    mse_bc_psi = tf.reduce_mean(tf.square(psi_t))+\
                 tf.reduce_mean(tf.square(psi_b))+\
                 tf.reduce_mean(tf.square(psi_l))+\
                 tf.reduce_mean(tf.square(psi_r))
    return mse_sum, mse_eq_phi, mse_eq_psi, mse_bc_t, mse_bc_steady, mse_bc_psi

###### function to compute the moment
def Moment(phi, psi, phi1):
    # theta: Nvy, Nvx
    # mu: Nvy, Nvx
    phi = phi + phi1
    d = d_mu * d_theta  # Nvy, Nvx
    times = phi * d   # -1, Nvy, Nvx
    rho = tf.reduce_sum(tf.reduce_sum(times, axis=2), axis=1)  # -1
    u_x = tf.reduce_sum(tf.reduce_sum(times * mu * np.cos(theta), axis=2), axis=1)
    u_y = tf.reduce_sum(tf.reduce_sum(times * mu * np.sin(theta), axis=2), axis=1)
    tau = tf.reduce_sum(tf.reduce_sum(2 / 3 * ((mu ** 2 - 1) * phi + psi) * d, axis=2), axis=1)

    return rho, u_x, u_y, tau

###### function to compute the loss for Equation
def Eq_comp(phi_x, phi_y, psi_x, psi_y, psi, phi, phi1):
    
    # calculate the Moment
    rho, u_x, u_y, tau = Moment(phi, psi, phi1)
    rho = tf.reshape(tf.tile(tf.reshape(rho, shape=(-1,1)), [1, Nvx*Nvy]), shape=(-1, Nvy, Nvx))    # -1, Nvy, Nvx
    u_x = tf.reshape(tf.tile(tf.reshape(u_x, shape=(-1,1)), [1, Nvx*Nvy]), shape=(-1, Nvy, Nvx))
    u_y = tf.reshape(tf.tile(tf.reshape(u_y, shape=(-1,1)), [1, Nvx*Nvy]), shape=(-1, Nvy, Nvx))
    tau = tf.reshape(tf.tile(tf.reshape(tau, shape=(-1,1)), [1, Nvx*Nvy]), shape=(-1, Nvy, Nvx))

    D = mu * (np.cos(theta) * phi_x + np.sin(theta) * phi_y)  # -1, Nvy, Nvx
    D = tf.cast(D, tf.float32)
    D_psi = mu * (np.cos(theta) * psi_x + np.sin(theta) * psi_y)  # -1, Nvy, Nvx
    D_psi = tf.cast(D_psi, tf.float32)

    I = delta*(rho+2*mu*(np.cos(theta)*u_x+np.sin(theta)*u_y)+tau*(mu**2-1))  # -1, Nvy, Nvx
    Eq_phi = D+delta*phi-I

    Eq_psi = D_psi+delta*psi-delta*tau/2
    return Eq_phi, Eq_psi

###### function to compute the loss for boundary condition
def Bc(phi_t, psi_t, phi_b, psi_b, phi_l, psi_l, phi_r, psi_r):
    u_t, u_b, u_l, u_r = rho(phi_t, phi_b, phi_l, phi_r)
    phi_t = phi_t*inp1-u_t
    phi_b = phi_b*inp2-u_b
    phi_l = phi_l*inp3-u_l
    phi_r = phi_r*inp4-u_r
    psi_t = psi_t*inp1
    psi_b = psi_b*inp2
    psi_l = psi_l*inp3
    psi_r = psi_r*inp4
    return phi_t, phi_b, phi_l, phi_r, psi_t, psi_b, psi_l, psi_r

###### boundary condition for train in the loss
def rho(phit, phib, phil, phir):
    phit = phit+phi1t
    phib = phib+phi1b
    phil = phil+phi1l
    phir = phir+phi1r
    mul_mu = mu * d_mu * d_theta * np.sqrt(np.pi)
    rho1 = 2 * tf.reduce_sum(
        tf.reduce_sum(phit * mul_mu * np.sin(theta) * inp2, axis=2),
        axis=1)
    rho1 = tf.reshape(tf.tile(tf.reshape(rho1, shape=[-1, 1]), [1, Nvx*Nvy]), shape=(-1, Nvy, Nvx))*inp1
    rho2 = -2 * tf.reduce_sum(
        tf.reduce_sum(phib * mul_mu * np.sin(theta) * inp1, axis=2),
        axis=1)
    rho2 = tf.reshape(tf.tile(tf.reshape(rho2, shape=[-1, 1]), [1, Nvx*Nvy]), shape=(-1, Nvy, Nvx))*inp2
    rho3 = -2 * tf.reduce_sum(
        tf.reduce_sum(phil * mul_mu * np.cos(theta) * inp4, axis=2),
        axis=1)
    rho3 = tf.reshape(tf.tile(tf.reshape(rho3, shape=[-1, 1]), [1, Nvx*Nvy]), shape=(-1, Nvy, Nvx))*inp3
    rho4 = 2 * tf.reduce_sum(
        tf.reduce_sum(phir * mul_mu * np.cos(theta) * inp3, axis=2),
        axis=1)  # -1,
    rho4 = tf.reshape(tf.tile(tf.reshape(rho4, shape=[-1, 1]), [1, Nvx*Nvy]), shape=(-1, Nvy, Nvx))*inp4
    return rho1, rho2, rho3, rho4

@tf.function
def grad(x_batch, xbl, xbr, xbt, xbb, phi1):
    with tf.GradientTape(persistent=True) as tape:
        mse_sum, mse_eq_phi, mse_eq_psi, mse_bc_t, mse_bc_steady, mse_bc_psi = loss(x_batch, xbl, xbr, xbt, xbb, phi1)
        grads = tape.gradient(mse_sum, Phi_model.trainable_variables)
        grads_ = tape.gradient(mse_sum, Psi_model.trainable_variables)
    return mse_sum, mse_eq_phi, mse_eq_psi, mse_bc_t, mse_bc_steady, mse_bc_psi, grads, grads_

@tf.function
def var_J(x_batch, xbl, xbr, xbt, xbb, phi1):
    kernel_size = 80
    L = len(layer_sizes)-1
    J_phi = []
    J_psi = []
    J_phi_bct = []
    J_phi_bc = []
    J_psi_bc = []
    with tf.GradientTape(persistent=True) as tape:
        mse_sum, mse_eq_phi, mse_eq_psi, mse_bc_t, mse_bc_steady, mse_bc_psi = loss(x_batch, xbl, xbr, xbt, xbb, phi1)
 
    for i in range(L*2):
        J_0 = tape.gradient(mse_eq_phi, Phi_model.trainable_variables[0])
        J_phi.append(J_0)

        J_0 = tape.gradient(mse_eq_psi, Psi_model.trainable_variables[0])
        J_psi.append(J_0)

        J_0 = tape.gradient(mse_bc_t, Phi_model.trainable_variables[0])
        J_phi_bct.append(J_0)

        J_0 = tape.gradient(mse_bc_steady, Phi_model.trainable_variables[0])
        J_phi_bc.append(J_0)

        J_0 = tape.gradient(mse_bc_psi, Psi_model.trainable_variables[0])
        J_psi_bc.append(J_0)
    
    N = len(J_phi)
    K_phi = []
    k_psi = []
    k_phi_bct = []
    k_phi_bc = []
    k_psi_bc = []
    for i in range(N):
        K_0 = tf.matmul(tf.reshape(J_phi[i], (kernel_size, -1)), tf.transpose(tf.reshape(J_phi[i], (kernel_size, -1))))
        K_phi.append(K_0)

        K_0 = tf.matmul(tf.reshape(J_psi[i], (kernel_size, -1)), tf.transpose(tf.reshape(J_psi[i], (kernel_size, -1))))
        k_psi.append(K_0)

        K_0 = tf.matmul(tf.reshape(J_phi_bct[i], (kernel_size, -1)), tf.transpose(tf.reshape(J_phi_bct[i], (kernel_size, -1))))
        k_phi_bct.append(K_0)

        K_0 = tf.matmul(tf.reshape(J_phi_bc[i], (kernel_size, -1)), tf.transpose(tf.reshape(J_phi_bc[i], (kernel_size, -1))))
        k_phi_bc.append(K_0)

        K_0 = tf.matmul(tf.reshape(J_psi_bc[i], (kernel_size, -1)), tf.transpose(tf.reshape(J_psi_bc[i], (kernel_size, -1))))
        k_psi_bc.append(K_0)
    K_phi = tf.reduce_sum(K_phi, axis=0)
    k_psi = tf.reduce_sum(k_psi, axis=0)
    k_phi_bct = tf.reduce_sum(k_phi_bct, axis=0)
    k_phi_bc = tf.reduce_sum(k_phi_bc, axis=0)
    k_psi_bc = tf.reduce_sum(k_psi_bc, axis=0)
    # ipdb.set_trace()
    return K_phi, k_psi, k_phi_bct, k_phi_bc, k_psi_bc

def fit(x_train, xbl, xbr, xbt, xbb, tf_iter, phi1):
    global phi_w
    global psi_w
    global phi_t_w
    global phi_b_w
    global phi_l_w
    global phi_r_w
    global psi_t_w
    global psi_b_w
    global psi_l_w
    global psi_r_w
    # Built in support for mini-batch, set to N_f (i.e. full batch) by default
    batch_sz = 50
    N_train = len(X_data)
    n_batches = N_train // batch_sz 
    start_time = time.time()
    tf_optimizer = tf.keras.optimizers.Adam(lr = 1e-4, beta_1=.90)
    Loss_sum=[]
    Loss_eq_phi = []
    Loss_eq_psi = []
    Loss_bc_phit = []
    Loss_bc_steady = []
    Loss_bc_psi = []

    log_weight = []

    print("starting Adam training: delta={}".format(delta))

    for epoch in range(tf_iter):
        for i in range(n_batches):
            idx = np.random.choice(N_train, min(batch_sz, N_train))
            idx_data = tf.convert_to_tensor(idx)
            x_batch = x_train[idx_data, :]
            x_batch = tf.convert_to_tensor(x_batch)
            phi1_batch = phi1[idx,...].astype(np.float32)

            mse_sum, mse_eq_phi, mse_eq_psi, mse_bc_t, mse_bc_steady, mse_bc_psi, grads, grads_ = grad(x_batch, xbl, xbr, xbt, xbb, phi1_batch)

            tf_optimizer.apply_gradients(zip(grads, Phi_model.trainable_variables))
            tf_optimizer.apply_gradients(zip(grads_, Psi_model.trainable_variables))
            # tf_optimizer_steady.apply_gradients(zip([-grads_phi_l, -grads_phi_b, -grads_phi_r, -grads_phi_t, -grads_psi_l, -grads_psi_b, -grads_psi_r, -grads_psi_t, -grads_phi, -grads_psi], [phi_l_w, phi_b_w, phi_r_w, phi_t_w, psi_l_w, psi_b_w, psi_r_w, psi_t_w, phi_w, psi_w]))

        if epoch % 10 == 0:

            K_phi, k_psi, k_phi_bct, k_phi_bc, k_psi_bc = var_J(x_batch, xbl, xbr, xbt, xbb, phi1_batch)

            K_trace = np.trace(K_phi+k_psi+k_phi_bct+k_phi_bc+k_psi_bc)
            phi_w = K_trace/np.trace(K_phi)
            psi_w = K_trace/np.trace(k_psi)
            phi_t_w = K_trace/np.trace(k_phi_bct)
            phi_l_w = K_trace/np.trace(k_phi_bc)
            phi_r_w = phi_l_w
            phi_b_w = phi_l_w
            psi_l_w = K_trace/np.trace(k_psi_bc)
            psi_r_w = psi_l_w
            psi_t_w = psi_l_w
            psi_b_w = psi_l_w

            elapsed = time.time() - start_time
            print('Epoch: %d, Time: %.2f' % (epoch, elapsed))
            print('Loss: %.4e,Eq_phi: %.4e,Eq_psi: %.4e,Bc_moving: %.4e,Bc_steady: %.4e,Bc_psi: %.4e' % (mse_sum, mse_eq_phi, mse_eq_psi, mse_bc_t, mse_bc_steady, mse_bc_psi))

            Loss_eq_phi.append(mse_eq_phi)
            Loss_eq_psi.append(mse_eq_psi)
            Loss_bc_phit.append(mse_bc_t)
            Loss_bc_steady.append(mse_bc_steady)
            Loss_bc_psi.append(mse_bc_psi)
            Loss_sum.append(mse_sum)

            log_weight.append(phi_w)
            log_weight.append(psi_w)
            log_weight.append(phi_t_w)
            log_weight.append(phi_b_w)
            log_weight.append(phi_l_w)
            log_weight.append(phi_r_w)
            log_weight.append(psi_t_w)
            log_weight.append(psi_b_w)
            log_weight.append(psi_l_w)
            log_weight.append(psi_r_w)

            start_time = time.time()
            if len(Loss_sum)>1:
                if mse_sum<np.array(Loss_sum)[:-1].min():
                    Phi_model.save('./Best_Net/SA_double_net_phi_model_test')
                    Psi_model.save('./Best_Net/SA_double_net_psi_model_test')
        if epoch%100==0:
            eigen_plot(K_phi, k_psi, k_phi_bct, k_phi_bc, k_psi_bc, epoch)
            plt.savefig("./fig/eigen_value(n={}).png".format(epoch), dpi=700)
    Loss_eq_phi = np.array(Loss_eq_phi)[:, None]
    Loss_eq_psi = np.array(Loss_eq_psi)[:, None]
    Loss_bc_phit = np.array(Loss_bc_phit)[:, None]
    Loss_bc_steady = np.array(Loss_bc_steady)[:, None]
    Loss_bc_psi = np.array(Loss_bc_psi)[:, None]
    Loss_sum = np.array(Loss_sum)[:, None]
    Loss = np.hstack((Loss_sum,Loss_eq_phi, Loss_eq_psi,Loss_bc_phit,Loss_bc_steady,Loss_bc_psi))
    np.savetxt('Loss.txt',Loss)

    log_weight = np.array(log_weight).reshape((-1, 10))
    np.savetxt('Log_weight.txt', log_weight)
    

def eigen_plot(K_phi, K_psi, K_phi_bct, K_phi_bc, K_psi_bc, n):
    # Compute eigenvalues
    lam_K_phi, _ = np.linalg.eig(K_phi)
    lam_K_psi, _ = np.linalg.eig(K_psi)
    lam_K_phit, _ = np.linalg.eig(K_phi_bct)
    lam_K_phi_bc, _ = np.linalg.eig(K_phi_bc)
    lam_K_psi_bc, _ = np.linalg.eig(K_psi_bc)
    # Sort in descresing order
    lam_K_phi = np.sort(np.real(lam_K_phi))[::-1]
    lam_K_psi = np.sort(np.real(lam_K_psi))[::-1]
    lam_K_phit = np.sort(np.real(lam_K_phit))[::-1]
    lam_K_phi_bc = np.sort(np.real(lam_K_phi_bc))[::-1]
    lam_K_psi_bc = np.sort(np.real(lam_K_psi_bc))[::-1]

    fig, axs = plt.subplots(1,5,figsize=(18, 5))
    axs = axs.ravel()

    axs[0].plot(lam_K_phi, label = 'n={}'.format(n))
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    axs[1].plot(lam_K_psi, label = 'n={}'.format(n))
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    axs[2].plot(lam_K_phit, label = 'n={}'.format(n))
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')

    axs[3].plot(lam_K_phi_bc, label = 'n={}'.format(n))
    axs[3].set_xscale('log')
    axs[3].set_yscale('log')

    axs[4].plot(lam_K_psi_bc, label = 'n={}'.format(n))
    axs[4].set_xscale('log')
    axs[4].set_yscale('log')
    # plt.show()

###### predict for the training domain
def predict(x_test, phi1_pre):
    x_test = np.split(x_test, 81, axis=0)
    Phi_pre_s, Psi_pre_s, Phi_x_pre_s, Phi_y_pre_s, Psi_x_pre_s, Psi_y_pre_s = net_domain(x_test[0])
    for i in range(1, 81):
        Phi_pre, Psi_pre, Phi_x_pre, Phi_y_pre, Psi_x_pre, Psi_y_pre = net_domain(x_test[i])
        Phi_pre_s = np.concatenate((Phi_pre_s,Phi_pre), axis=0, dtype=np.float32)
        Psi_pre_s = np.concatenate((Psi_pre_s,Psi_pre), axis=0, dtype=np.float32)
        Phi_x_pre_s = np.concatenate((Phi_x_pre_s,Phi_x_pre), axis=0, dtype=np.float32)
        Phi_y_pre_s = np.concatenate((Phi_y_pre_s,Phi_y_pre), axis=0, dtype=np.float32)
        Psi_x_pre_s = np.concatenate((Psi_x_pre_s,Psi_x_pre), axis=0, dtype=np.float32)
        Psi_y_pre_s = np.concatenate((Psi_y_pre_s,Psi_y_pre), axis=0, dtype=np.float32)

    phi1_pre = phi1_pre.astype(np.float32)
    rho, ux, uy, tau = Moment(Phi_pre_s, Psi_pre_s, phi1_pre)
    rho = rho.numpy()
    ux = ux.numpy()
    uy = uy.numpy()
    tau = tau.numpy()
    # ipdb.set_trace()
    Eq_phi, Eq_psi = Eq_comp(Phi_x_pre_s, Phi_y_pre_s, Psi_x_pre_s, Psi_y_pre_s, Psi_pre_s, Phi_pre_s, phi1_pre)
    Eq_phi = Eq_phi.numpy()
    Eq_psi = Eq_psi.numpy()

    return Phi_pre_s, Psi_pre_s, rho, ux, uy, tau, Eq_phi, Eq_psi

###### plot ######
def plot(Phi_pre, Psi_pre, rho, ux, uy, tau, Eq_phi, Eq_Psi):
    # plot
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(11, 15))

    levels = np.linspace(Phi_pre[0, ...].min(), Phi_pre[0, ...].max(), 100)
    Phi_fig = ax1.contourf(mu, theta, Phi_pre[0, ...], cmap='jet', levels=levels)
    fig.colorbar(Phi_fig, ax=ax1)
    ax1.set(title='$\phi$', xlabel='$\mu$', ylabel='$\\theta$')

    # plot D
    levels = np.linspace(ux.min(), ux.max(), 20)
    ux_fig = ax2.contourf(xx, yy, ux.reshape(Nx, Ny), cmap='jet', levels=levels)
    fig.colorbar(ux_fig, ax=ax2)
    ax2.set(title='$u_x$', xlabel='$x$', ylabel='$y$')

    levels = np.linspace(uy.min(), uy.max(), 20)
    uy_fig = ax5.contourf(xx, yy, uy.reshape(Nx, Ny), cmap='jet', levels=levels)
    fig.colorbar(uy_fig, ax=ax5)
    ax5.set(title='$u_y$', xlabel='$x$', ylabel='$y$')

    levels = np.linspace(tau.min(), tau.max(), 12)
    tau_fig = ax6.contourf(xx, yy, tau.reshape(Nx, Ny), cmap='jet', levels=levels)
    fig.colorbar(tau_fig, ax=ax6)
    ax6.set(title='$\\tau$', xlabel='$x$', ylabel='$y$')

    # predict for Eq
    levels = np.linspace(np.abs(Eq_phi[0, ...]).min(), np.abs(Eq_phi[0, ...]).max(), 100)
    Eq_fig = ax3.contourf(mu, theta, np.abs(Eq_phi[0, ...]), cmap='jet', levels = levels)
    fig.colorbar(Eq_fig, ax=ax3)
    ax3.set(title='$Eq_\phi$', xlabel='$\mu$', ylabel='$\\theta$')

    # stream plot of velocity
    # ax4.contourf(xx, yy, rho.reshape(Nx, Ny), cmap='jet')
    # plt.colorbar()
    ax4.streamplot(X_t[:,0].reshape(Nx,Ny), X_t[:,1].reshape(Nx,Ny), ux.reshape(Nx, Ny), uy.reshape(Nx, Ny), density=0.7, color='k')
    ax4.set(title='streamline of velocity', xlabel='$x$', ylabel='$y$')
    ax4.set_ylim(0,1.)
    ax4.set_xlim(0,1.) 
    # plt.show()
    plt.tick_params(labelsize=13) 
    plt.savefig("./fig/SA_double_net_batch100_delta_{}_theta_200.png".format(delta), dpi=700)

# batch_sz
# main
#########################################################
################self adaptive weights####################
#########################################################

phi_b_w = 20.
phi_l_w = 10.
phi_t_w = 10.
phi_r_w = 10.
psi_t_w = 1.
psi_l_w = 1.
psi_b_w = 1.
psi_r_w = 1.
phi_w   = 10.
psi_w   = 1.

# moving_weights = tf.Variable(tf.random.uniform([Nxb, Nvy, Nvx]))
##########################################
##########Train and Predict###############
##########################################
UseLoadModel = False
if UseLoadModel:
    Phi_model = keras.models.load_model('./Best_Net/SA_double_net_phi_model_{}'.format(delta))
    Psi_model = keras.models.load_model('./Best_Net/SA_double_net_psi_model_{}'.format(delta))
else: 
  UseLoadModel_train = False
  if UseLoadModel_train:
      Phi_model = keras.models.load_model('./Best_Net/SA_double_net_phi_model_test')
      Psi_model = keras.models.load_model('./Best_Net/SA_double_net_psi_model_test')
      Phi_model.trainable=False
      Psi_model.trainable=False
      Phi_model.layers[-1].trainable = True
      Psi_model.layers[-1].trainable = True

      data = scipy.io.loadmat('./data/phi1_delta_{}_theta_{}.mat'.format(delta, Nvy))
      phi1 = np.array(data['phi1']).reshape(-1, Nvy, Nvx)

      data = scipy.io.loadmat('./data/phi1t_delta_{}_theta_{}.mat'.format(delta, Nvy))
      phi1t = np.array(data['phi1']).reshape(-1, Nvy, Nvx)
      data = scipy.io.loadmat('./data/phi1b_delta_{}_theta_{}.mat'.format(delta, Nvy))
      phi1b = np.array(data['phi1']).reshape(-1, Nvy, Nvx)
      data = scipy.io.loadmat('./data/phi1l_delta_{}_theta_{}.mat'.format(delta, Nvy))
      phi1l = np.array(data['phi1']).reshape(-1, Nvy, Nvx)
      data = scipy.io.loadmat('./data/phi1r_delta_{}_theta_{}.mat'.format(delta, Nvy))
      phi1r = np.array(data['phi1']).reshape(-1, Nvy, Nvx)
      fit(X_data, Xbl, Xbr, Xbt, Xbb, tf_iter = 500, phi1=phi1)


  else:
    ## load data
      phi1_cul()
      data = scipy.io.loadmat('./data/phi1_delta_{}_theta_{}.mat'.format(delta, Nvy))
      phi1 = np.array(data['phi1']).reshape(-1, Nvy, Nvx)

      data = scipy.io.loadmat('./data/phi1t_delta_{}_theta_{}.mat'.format(delta, Nvy))
      phi1t = np.array(data['phi1']).reshape(-1, Nvy, Nvx)
      data = scipy.io.loadmat('./data/phi1b_delta_{}_theta_{}.mat'.format(delta, Nvy))
      phi1b = np.array(data['phi1']).reshape(-1, Nvy, Nvx)
      data = scipy.io.loadmat('./data/phi1l_delta_{}_theta_{}.mat'.format(delta, Nvy))
      phi1l = np.array(data['phi1']).reshape(-1, Nvy, Nvx)
      data = scipy.io.loadmat('./data/phi1r_delta_{}_theta_{}.mat'.format(delta, Nvy))
      phi1r = np.array(data['phi1']).reshape(-1, Nvy, Nvx)

      Phi_model = neural_net(layer_sizes)
      Phi_model.summary()
      Psi_model = neural_net(layer_sizes)
      Psi_model.summary()
      tf_iter=20000
      fit(X_data, Xbl, Xbr, Xbt, Xbb, tf_iter = tf_iter, phi1=phi1)

data = scipy.io.loadmat('./data/phi1_pre_delta_{}_theta_{}.mat'.format(delta, Nvy))
phi1_pre = np.array(data['phi1']).reshape(-1, Nvy, Nvx)
Phi_pre, Psi_pre, rho, ux, uy, tau, Eq_phi, Eq_psi = predict(X_t, phi1_pre)
scipy.io.savemat('./data/uy_delta_{}_theta_{}_PINN_{}.mat'.format(delta, Nvy, tf_iter), {'uy':uy})
scipy.io.savemat('./data/ux_delta_{}_theta_{}_PINN_{}.mat'.format(delta, Nvy, tf_iter), {'ux':ux})
scipy.io.savemat('./data/Phi_pre_delta_{}_theta_{}_PINN_{}.mat'.format(delta, Nvy, tf_iter), {'Phi_pre':Phi_pre+phi1_pre})

plot(Phi_pre, Psi_pre, rho, ux, uy, tau, Eq_phi, Eq_psi)