import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from hyper_parameters import global_var, X_b, X_test1, Dh

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

np.random.seed(42)
tf.random.set_seed(42)

### import data 
Nvx = global_var.get_value('Nvx')
Nvy = global_var.get_value('Nvy')
X_train = global_var.get_value('X_train')
mu = global_var.get_value('mu')
theta = global_var.get_value('theta')
d_mu = global_var.get_value('d_mu')
d_theta = global_var.get_value('d_theta')
X_test = global_var.get_value('X_test')
delta = global_var.get_value('delta')
top = global_var.get_value('top')
bottom = global_var.get_value('bottom')
left = global_var.get_value('left')
right = global_var.get_value('right')
edge = global_var.get_value('edge')
Nxb = global_var.get_value('Nxb')
mu = mu.astype(np.float32)
theta = theta.astype(np.float32)
d_mu = d_mu.astype(np.float32)
d_theta = d_theta.astype(np.float32)

N_l = 4
N_neur = 40

activation_function = tf.nn.sigmoid
a_f = 'sigmoid'
layer_sizes=[2]+[N_neur]*N_l+[Nvx*Nvy]

def neural_net(layer_sizes):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    for i, width in enumerate(layer_sizes[1:-1]):
        model.add(layers.Dense(width, 
                               activation=activation_function,
                               kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/(layer_sizes[i+1] + layer_sizes[i+2])), seed=None),
                               bias_initializer="zeros"))
    model.add(layers.Dense(layer_sizes[-1], 
                           activation=None,
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=np.sqrt(2/(layer_sizes[i+1] + layer_sizes[i+2])), seed=None),
                           bias_initializer="zeros"))
    return model

@tf.function
def net_domain(x):   
    scale = (X_train.max(axis=0) - X_train.min(axis=0))
    # x = 2.0 * (x - X_train.min(axis=0)) / scale - 1.0 
    # with tf.GradientTape(persistent=True) as g:
    #     g.watch(x)
    #     phi = Phi_model(x) # Nx*Ny, Nvx*Nvy
    # # grad for phi1
    # phi_xy = g.batch_jacobian(phi, x)
    # phi_x = tf.reshape(phi_xy[..., 0], shape=(-1, Nvy, Nvx))*2/scale[0]
    # phi_y = tf.reshape(phi_xy[..., 1], shape=(-1, Nvy, Nvx))*2/scale[1]

    x = (x - X_train.min(axis=0)) / scale
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        phi = Phi_model(x) # Nx*Ny, Nvx*Nvy
    # grad for phi1
    phi_xy = g.batch_jacobian(phi, x)
    phi_x = tf.reshape(phi_xy[..., 0], shape=(-1, Nvy, Nvx))/scale[0]
    phi_y = tf.reshape(phi_xy[..., 1], shape=(-1, Nvy, Nvx))/scale[1]
    return tf.reshape(phi, shape=(-1, Nvy, Nvx)), phi_x, phi_y

@tf.function
def net_Phi_bc(x):
    scale = (X_train.max(axis=0) - X_train.min(axis=0))
    x = 2.0 * (x - X_train.min(axis=0)) / scale - 1.0 
    phi = Phi_model(x) # Nx*Ny, Nvx*Nvy
    return tf.reshape(phi, shape=(-1, Nvy, Nvx))

### loss ###
def loss(X_batch, X_b):
    # equation
    Phi_pre, Phi_x_pre, Phi_y_pre = net_domain(X_batch)
    Eq_phi = Eq_comp(Phi_x_pre, Phi_y_pre, Phi_pre)
    # boundary
    X_left = X_b['Xb_left']
    X_bot = X_b['Xb_bot']
    X_edge = X_b['Xb_edge']

    phi_edge = net_Phi_bc(X_edge)
    phi_bot  = net_Phi_bc(X_bot)
    phi_left = net_Phi_bc(X_left)

    Bc_edge, Bc_bot, Bc_left = Bc(phi_edge, phi_bot, phi_left)

    mse_eq_phi = tf.reduce_mean(tf.square(Eq_phi))
    mse_bc_edge = tf.reduce_mean(tf.square(Bc_edge))*10
    mse_bc_bot = tf.reduce_mean(tf.square(Bc_bot))*10
    mse_bc_left = tf.reduce_mean(tf.square(Bc_left))*10
    mse_sum = mse_eq_phi+mse_bc_edge+mse_bc_bot+mse_bc_left

    # ipdb.set_trace()
    return mse_sum, mse_eq_phi, mse_bc_edge, mse_bc_bot, mse_bc_left

###### function to compute the moment
def Moment(phi):
    # theta: Nvy, Nvx
    # mu: Nvy, Nvx
    u = tf.reduce_sum(tf.reduce_sum(phi * d_mu * d_theta, axis=2), axis=1)

    return u

###### function to compute the loss for Equation
def Eq_comp(phi_x, phi_y, phi):
    
    # calculate the Moment
    u = Moment(phi)
    u = tf.reshape(tf.tile(tf.reshape(u, shape=(-1,1)), [1, Nvx*Nvy]), shape=(-1, Nvy, Nvx))

    D = mu * (np.cos(theta) * phi_x + np.sin(theta) * phi_y)  # -1, Nvy, Nvx
    D = tf.cast(D, tf.float32)

    I = delta*u-1/2  # -1, Nvy, Nvx
    Eq_phi = D+delta*phi-I
    return Eq_phi

###### function to compute the loss for boundary condition
def Bc(phi_edge, phi_bot, phi_left):

    Bc_edge = phi_edge*edge
    Bc_bot  = phi_bot*bottom
    Bc_left = phi_left*left
    return Bc_edge, Bc_bot, Bc_left 

@tf.function
def grad(X_batch, X_b):
    with tf.GradientTape(persistent=True) as tape:
        mse_sum, mse_eq_phi, mse_bc_edge, mse_bc_bot, mse_bc_left = loss(X_batch, X_b)
        grads_phi = tape.gradient(mse_sum, Phi_model.trainable_variables)
    return mse_sum, mse_eq_phi, mse_bc_edge, mse_bc_bot, mse_bc_left, grads_phi

# @tf.function
# def var_J(X_batch, X_b):
#     kernel_size = 100
#     L = len(layer_sizes)-1
#     J_phi = []
#     J_psi = []
#     J_phi_top = []
#     J_phi_hor = []
#     J_phi_ver = []
#     J_psi_bc = []
#     with tf.GradientTape(persistent=True) as tape:
#         mse_sum, mse_eq_phi, mse_eq_psi, mse_bc_top, mse_bc_ver, mse_bc_hor, mse_bc_psi, mse_peri = loss(X_batch, X_b)
 
#     for i in range(L*2):
#         J_0 = tape.gradient(mse_eq_phi, Phi_model.trainable_variables[0])
#         J_phi.append(J_0)

#         J_0 = tape.gradient(mse_eq_psi, Psi_model.trainable_variables[0])
#         J_psi.append(J_0)

#         J_0 = tape.gradient(mse_bc_top, Phi_model.trainable_variables[0])
#         J_phi_top.append(J_0)

#         J_0 = tape.gradient(mse_bc_hor, Phi_model.trainable_variables[0])
#         J_phi_hor.append(J_0)

#         J_0 = tape.gradient(mse_bc_ver, Phi_model.trainable_variables[0])
#         J_phi_ver.append(J_0)

#         J_0 = tape.gradient(mse_bc_psi, Psi_model.trainable_variables[0])
#         J_psi_bc.append(J_0)
    
#     N = len(J_phi)
#     K_phi = []
#     K_psi = []
#     K_phi_top = []
#     K_phi_hor = []
#     K_phi_ver = []
#     K_psi_bc = []
#     for i in range(N):
#         K_0 = tf.matmul(tf.reshape(J_phi[i], (kernel_size, -1)), tf.transpose(tf.reshape(J_phi[i], (kernel_size, -1))))
#         K_phi.append(K_0)

#         K_0 = tf.matmul(tf.reshape(J_psi[i], (kernel_size, -1)), tf.transpose(tf.reshape(J_psi[i], (kernel_size, -1))))
#         K_psi.append(K_0)

#         K_0 = tf.matmul(tf.reshape(J_phi_top[i], (kernel_size, -1)), tf.transpose(tf.reshape(J_phi_top[i], (kernel_size, -1))))
#         K_phi_top.append(K_0)

#         K_0 = tf.matmul(tf.reshape(J_phi_hor[i], (kernel_size, -1)), tf.transpose(tf.reshape(J_phi_hor[i], (kernel_size, -1))))
#         K_phi_hor.append(K_0)

#         K_0 = tf.matmul(tf.reshape(J_phi_ver[i], (kernel_size, -1)), tf.transpose(tf.reshape(J_phi_ver[i], (kernel_size, -1))))
#         K_phi_ver.append(K_0)

#         K_0 = tf.matmul(tf.reshape(J_psi_bc[i], (kernel_size, -1)), tf.transpose(tf.reshape(J_psi_bc[i], (kernel_size, -1))))
#         K_psi_bc.append(K_0)

#     K_phi = tf.reduce_sum(K_phi, axis=0)
#     K_psi = tf.reduce_sum(K_psi, axis=0)
#     K_phi_top = tf.reduce_sum(K_phi_top, axis=0)
#     K_phi_hor = tf.reduce_sum(K_phi_hor, axis=0)
#     K_phi_ver = tf.reduce_sum(K_phi_ver, axis=0)
#     K_psi_bc = tf.reduce_sum(K_psi_bc, axis=0)
#     # ipdb.set_trace()
#     return K_phi, K_psi, K_phi_top, K_phi_hor, K_phi_ver, K_psi_bc

def fit(X_train, X_b, tf_iter):
    # global phi_w
    # global psi_w
    # global phi_t_w
    # global phi_b_w
    # global phi_l_w
    # global phi_r_w
    # global psi_t_w
    # global psi_b_w
    # global psi_l_w
    # global psi_r_w
    # Built in support for mini-batch, set to N_f (i.e. full batch) by default
    batch_sz = 50
    N_train = len(X_train)
    n_batches = N_train // batch_sz 
    start_time = time.time()
    tf_optimizer = tf.keras.optimizers.Adam(lr = 1e-4, beta_1=.90)
    # Loss log
    Loss_sum    = []
    Loss_eq_phi = []
    Loss_left   = []
    Loss_bot    = []
    Loss_edge   = []

    log_weight = []

    print("starting Adam training")

    for epoch in range(tf_iter):
        for i in range(n_batches):
            idx = np.random.choice(N_train, min(batch_sz, N_train))
            idx_data = tf.convert_to_tensor(idx)
            x_batch = X_train[idx_data, :]
            x_batch = tf.convert_to_tensor(x_batch)

            mse_sum, mse_eq_phi, mse_bc_edge, mse_bc_bot, mse_bc_left, grads_phi = grad(x_batch, X_b)

            tf_optimizer.apply_gradients(zip(grads_phi, Phi_model.trainable_variables))

        if epoch % 10 == 0:

            # K_phi, K_psi, K_phi_top, K_phi_hor, K_phi_ver, K_psi_bc = var_J(x_batch, X_b)

            # K_trace = np.trace(K_phi+K_psi+K_phi_top+K_phi_hor+K_phi_ver+K_psi_bc)
            # phi_w   = K_trace/np.trace(K_phi)
            # psi_w   = K_trace/np.trace(K_psi)
            # phi_t_w = K_trace/np.trace(K_phi_top)
            # phi_l_w = K_trace/np.trace(K_phi_ver)
            # phi_r_w = phi_l_w
            # phi_b_w = K_trace/np.trace(K_phi_hor)
            # psi_l_w = K_trace/np.trace(K_psi_bc)
            # psi_r_w = psi_l_w
            # psi_t_w = psi_l_w
            # psi_b_w = psi_l_w

            elapsed = time.time() - start_time
            print('Epoch: %d, Time: %.2f' % (epoch, elapsed))
            print('Loss: %.4e,Eq_phi: %.4e,Bc_edge: %.4e,Bc_bot: %.4e,Bc_left: %.4e' % (mse_sum, mse_eq_phi, mse_bc_edge, mse_bc_bot, mse_bc_left))

            Loss_sum.append(mse_sum)
            Loss_eq_phi.append(mse_eq_phi)
            Loss_left.append(mse_bc_left)
            Loss_bot.append(mse_bc_bot)
            Loss_edge.append(mse_bc_edge)

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
            if len(Loss_eq_phi)>1:
                if mse_sum<np.array(Loss_sum)[:-1].min():
                    Phi_model.save('./Best_Net/SA_double_net_phi_model_test')
        # if epoch%100==0:
        #     eigen_plot(K_phi, K_psi, K_phi_top, K_phi_hor, K_phi_ver, K_psi_bc, epoch)
        #     plt.savefig("./fig/eigen_value(n={}).png".format(epoch), dpi=700)
    Loss_eq_phi = np.array(Loss_eq_phi)[:, None]
    Loss_bot    = np.array(Loss_bot)[:, None]
    Loss_left   = np.array(Loss_left)[:, None]
    Loss_edge   = np.array(Loss_edge)[:, None]
    Loss_sum    = np.array(Loss_sum)[:, None]
    Loss = np.hstack((Loss_sum, Loss_eq_phi, Loss_bot, Loss_left,Loss_edge))
    np.savetxt('./data/Loss_{}_{}_{}.txt'.format(a_f, N_l, N_neur),Loss)

    # log_weight = np.array(log_weight).reshape((-1, 10))
    # np.savetxt('Log_weight.txt', log_weight)

def eigen_plot(K_phi, K_psi, K_phi_top, K_phi_hor, K_phi_ver, K_psi_bc, epoch):
    # Compute eigenvalues
    lam_K_phi, _ = np.linalg.eig(K_phi)
    lam_K_psi, _ = np.linalg.eig(K_psi)
    lam_K_top, _ = np.linalg.eig(K_phi_top)
    lam_K_hor, _ = np.linalg.eig(K_phi_hor)
    lam_K_ver, _ = np.linalg.eig(K_phi_ver)
    lam_K_psi_bc, _ = np.linalg.eig(K_psi_bc)
    # Sort in descresing order
    lam_K_phi = np.sort(np.real(lam_K_phi))[::-1]
    lam_K_psi = np.sort(np.real(lam_K_psi))[::-1]
    lam_K_top = np.sort(np.real(lam_K_top))[::-1]
    lam_K_hor = np.sort(np.real(lam_K_hor))[::-1]
    lam_K_ver = np.sort(np.real(lam_K_ver))[::-1]
    lam_K_psi_bc = np.sort(np.real(lam_K_psi_bc))[::-1]

    fig = plt.figure(figsize=(18, 5))

    plt.subplot(1,6,1)
    plt.plot(lam_K_phi, label = '$n={}$'.format(epoch))
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(1,6,2)
    plt.plot(lam_K_psi, label = '$n={}$'.format(epoch))
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(1,6,3)
    plt.plot(lam_K_top, label = '$n={}$'.format(epoch))
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(1,6,4)
    plt.plot(lam_K_ver, label = '$n={}$'.format(epoch))
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(1,6,5)
    plt.plot(lam_K_hor, label = '$n={}$'.format(epoch))
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(1,6,6)
    plt.plot(lam_K_psi_bc, label = '$n={}$'.format(epoch))
    plt.xscale('log')
    plt.yscale('log')
    # plt.show()

###### predict for the training domain
def predict(x_test):
    n_spli = 100
    x_test = np.split(x_test, n_spli, axis=0)
    Phi_pre_s, Phi_x_pre_s, Phi_y_pre_s = net_domain(x_test[0])
    for i in range(1, len(x_test)):
        Phi_pre, Phi_x_pre, Phi_y_pre = net_domain(x_test[i])
        Phi_pre_s = np.concatenate((Phi_pre_s,Phi_pre), axis=0, dtype=np.float32)
        Phi_x_pre_s = np.concatenate((Phi_x_pre_s,Phi_x_pre), axis=0, dtype=np.float32)
        Phi_y_pre_s = np.concatenate((Phi_y_pre_s,Phi_y_pre), axis=0, dtype=np.float32)

    u = Moment(Phi_pre_s)
    u = u.numpy()
    # ipdb.set_trace()
    Eq_phi = Eq_comp(Phi_x_pre_s, Phi_y_pre_s, Phi_pre_s)
    Eq_phi = Eq_phi.numpy()

    return Phi_pre_s, u, Eq_phi

###### plot ######
def plot(Phi_pre, u, Eq_phi):
    # plot
    fig, axs = plt.subplots(1,2,figsize=(15,7), constrained_layout=True)
    axs = axs.ravel()
    # eq
    im0 = axs[0].scatter(X_test[:, 0:1],X_test[:, 1:2],c=Eq_phi[:, 0,0],cmap=plt.cm.rainbow,vmin=min(Eq_phi[:, 0,0]),vmax=max(Eq_phi[:, 0,0]))
    axs[0].set_title(r'$residual\;of\; Equation\;\phi$')
    fig.colorbar(im0, ax=axs[0])

    # U
    u = -u
    u[(X_test[:, 0]+X_test[:, 1])*Dh>1] = np.nan
    im2 = axs[1].scatter(X_test[:, 0:1],X_test[:, 1:2],c=u.reshape(-1,100),cmap=plt.cm.rainbow,vmin=min(u),vmax=max(u))
    axs[1].set_title(r'$velocity\; contour$')
    fig.colorbar(im2, ax=axs[1])    
    for i in range(len(axs)):
      axs[i].set_xlabel('x', fontsize=13)
      axs[i].set_ylabel('y', fontsize=13)
      axs[i].set_xlim(0,X_test.max())
      axs[i].set_ylim(0,X_test.max())
      axs[i].tick_params(labelsize=13) 
    # plt.show()
    # fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.2, wspace=0.3)
    fig.savefig("./fig/SA_double_net_batch100_delta_{}_theta_200.png".format(delta), dpi=700)

# batch_sz
# main
#########################################################
################self adaptive weights####################
#########################################################

phi_b_w = 1. 
phi_l_w = 1. 
phi_t_w = 1. 
phi_r_w = 1. 
psi_t_w = 1. 
psi_l_w = 1. 
psi_b_w = 1. 
psi_r_w = 1. 
phi_w = 1. 
psi_w = 1. 

# moving_weights = tf.Variable(tf.random.uniform([Nxb, Nvy, Nvx]))
##########################################
##########Train and Predict###############
##########################################
UseLoadModel = False
UseLoadModeltoTrain = False
if UseLoadModel:
    Phi_model = keras.models.load_model('./Best_Net/SA_double_net_phi_model_test')
else:
  if UseLoadModeltoTrain:
    Phi_model = keras.models.load_model('./Best_Net/SA_double_net_phi_model_test')
    Phi_model = neural_net(layer_sizes)
    Phi_model.summary()
    fit(X_train, X_b, tf_iter = 2000)
  else:
    Phi_model = neural_net(layer_sizes)
    Phi_model.summary()
    fit(X_train, X_b, tf_iter = 5000)

Phi_pre, u, Eq_phi = predict(X_test)
scipy.io.savemat('./data/u_delta_{}_theta_{}_PINN.mat'.format(delta, Nvy), {'u':u})
scipy.io.savemat('./data/Phi_pre_delta_{}_PINN.mat'.format(delta), {'Phi_pre':Phi_pre})
plot(Phi_pre, u, Eq_phi)
Phi_pre, u, Eq_phi = predict(X_test1)
scipy.io.savemat('./data/u_symmetry_delta_{}_theta_{}_PINN.mat'.format(delta, Nvy), {'u':u})