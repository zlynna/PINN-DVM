import numpy as np
from hyper_parameters import global_var
import scipy.io

x = global_var.get_value('x')
y = global_var.get_value('y')
xb_ = global_var.get_value('xb')
x1 = global_var.get_value('x1')
y1 = global_var.get_value('y1')
mu = global_var.get_value('c')
theta = global_var.get_value('t')
delta = global_var.get_value('delta')
N_theta = global_var.get_value('Nvy')

def phi1_cul():
  # phi train
    phi= np.zeros((len(x), len(y), len(theta), len(mu)))
    for i in range(len(x)):
      for j in range(len(y)):
        for k in range(len(mu)):
          for l in range(len(theta)):
            b = y[j]-np.tan(theta[l])*x[i]
            if np.pi<theta[l]<np.pi*2:
              if 0<(1-b)/np.tan(theta[l])<1:
                phi[j,i,l,k]=2*mu[k]*np.cos(theta[l])*np.exp(-delta*(1-y[j])/abs(np.sin(theta[l]))/mu[k])
    scipy.io.savemat('./data/phi1_delta_{}_theta_{}.mat'.format(delta, N_theta), {'phi1':phi})
  # phi test
    phi= np.zeros((len(x1), len(y1), len(theta), len(mu)))
    for i in range(len(x1)):
      for j in range(len(y1)):
        for k in range(len(mu)):
          for l in range(len(theta)):
            b = y1[j]-np.tan(theta[l])*x1[i]
            if np.pi<theta[l]<np.pi*2:
              if 0<(1-b)/np.tan(theta[l])<1:
                phi[j,i,l,k]=2*mu[k]*np.cos(theta[l])*np.exp(-delta*(1-y1[j])/abs(np.sin(theta[l]))/mu[k])
    scipy.io.savemat('./data/phi1_pre_delta_{}_theta_{}.mat'.format(delta, N_theta), {'phi1':phi})

    # phi1t
    xb = xb_
    yb = np.ones_like(xb_)
    phi= np.zeros((len(xb), len(theta), len(mu)))
    for i in range(len(xb)):
      for k in range(len(mu)):
        for l in range(len(theta)):
          b = yb[i]-np.tan(theta[l])*xb[i]
          if np.pi<theta[l]<np.pi*2:
            if 0<(1-b)/np.tan(theta[l])<1:
              phi[i,l,k]=2*mu[k]*np.cos(theta[l])*np.exp(-delta*(1-yb[i])/abs(np.sin(theta[l]))/mu[k])
    scipy.io.savemat('./data/phi1t_delta_{}_theta_{}.mat'.format(delta, N_theta), {'phi1':phi})

    # phi1b
    xb = xb_
    yb = np.zeros_like(xb_)
    phi= np.zeros((len(xb), len(theta), len(mu)))
    for i in range(len(xb)):
      for k in range(len(mu)):
        for l in range(len(theta)):
          b = yb[i]-np.tan(theta[l])*xb[i]
          if np.pi<theta[l]<np.pi*2:
            if 0<(1-b)/np.tan(theta[l])<1:
              phi[i,l,k]=2*mu[k]*np.cos(theta[l])*np.exp(-delta*(1-yb[i])/abs(np.sin(theta[l]))/mu[k])
    scipy.io.savemat('./data/phi1b_delta_{}_theta_{}.mat'.format(delta, N_theta), {'phi1':phi})

    # phi1l
    xb = np.zeros_like(xb_)
    yb = xb_
    phi= np.zeros((len(xb), len(theta), len(mu)))
    for i in range(len(xb)):
      for k in range(len(mu)):
        for l in range(len(theta)):
          b = yb[i]-np.tan(theta[l])*xb[i]
          if np.pi<theta[l]<np.pi*2:
            if 0<(1-b)/np.tan(theta[l])<1:
              phi[i,l,k]=2*mu[k]*np.cos(theta[l])*np.exp(-delta*(1-yb[i])/abs(np.sin(theta[l]))/mu[k])
    scipy.io.savemat('./data/phi1l_delta_{}_theta_{}.mat'.format(delta, N_theta), {'phi1':phi})

    # phi1r
    xb = np.ones_like(xb_)
    yb = xb_
    phi= np.zeros((len(xb), len(theta), len(mu)))
    for i in range(len(xb)):
      for k in range(len(mu)):
        for l in range(len(theta)):
          b = yb[i]-np.tan(theta[l])*xb[i]
          if np.pi<theta[l]<np.pi*2:
            if 0<(1-b)/np.tan(theta[l])<1:
              phi[i,l,k]=2*mu[k]*np.cos(theta[l])*np.exp(-delta*(1-yb[i])/abs(np.sin(theta[l]))/mu[k])
    scipy.io.savemat('./data/phi1r_delta_{}_theta_{}.mat'.format(delta, N_theta), {'phi1':phi})

phi1_cul()
# # data = scipy.io.loadmat('./phi1_pre.mat')
# # phi = np.array(data['phi1'])
# # fig, ax = plt.subplots()
# # xx,yy = np.meshgrid(x,y)
# # levels = np.linspace(phi[..., -1,16].min(), phi[..., -1,16].max(), 200)
# # zs=ax.contourf(xx,yy,phi[:,:,-1,16])
# # fig.colorbar(zs, ax=ax)
# # plt.show()