import module as global_var
import numpy as np
import pandas as pd

# data generation
# initialization
# spatial grid
Lx = 1.
Ly = 1.
Nx = 101
Ny = 101
# x direction
# p = 1.1
# s = np.logspace(0, np.log10(p**((Ny-1)/2)), int((Ny-1)/2+1))
# s1 = 0.5*(s-s[0])/(max(s)-min(s))
# y = np.concatenate((s1[:-1], np.flip(1-s1)))
y = np.linspace(0,1,Ny)
# y direction
x = y
# meshgrid of xy
[xx, yy] = np.meshgrid(x, y)
X = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))

# velocity grid
Nvx = 8
Nvy = 200
delta = 1
v_max = 7
# c = v_max*(np.arange(-Nvx*2+1,Nvx*2-1,2)**3/(Nvx*2-1)**3)
# w = 2*v_max*(3*np.arange(-Nvx*2+1,Nvx*2-1,2)**2)/(Nvx*2-1)**3/np.pi*np.exp(-c**2)*c
# c = -c[:Nvx][::-1]
# w = -w[:Nvx][::-1]
# meshgrid
# [vxx, vyy] = np.meshgrid(vx, vy)
# n_i = np.linspace(-Nvx+1, Nvx, Nvx)
# n_j = np.linspace(-Nvx+1, Nvx, Nvx)
# [N_i, N_j] = np.meshgrid(n_i, n_j)
# velocity transpose
cw8 = pd.read_table('cw8.dat', sep=' ', header=None, engine='python')
cw8 = np.array(cw8)
c = cw8[:, 0:1]
w = cw8[:, 1:2]
# theta
t = np.arange(0, 2*np.pi, 2*np.pi/Nvy)
wt = []
for j in range(Nvy):
    if j/2-int(j/2)==0:
        wt_ = 4*np.pi/3/Nvy
    if j/2-int(j/2)!=0:
        wt_ = 8*np.pi/3/Nvy
    if j==0 or j==Nvy-1: wt_ = 1/3*2*np.pi/Nvy
    wt.append(wt_)
wt = np.array(wt)
mu, theta = np.meshgrid(c, t)
d_mu, d_theta = np.meshgrid(w, wt)

# d_theta = np.ones_like(d_theta)*2*np.pi/Nvy
# f0 = np.sum(mu*np.cos(theta)*d_mu*d_theta) # for test
# print(f0)
# raise ValueError

# theta for boundary conditions
inp1 = np.sin(theta)<0  # [pi, 2*pi]
inp2 = np.sin(theta)>0  # [0, pi]
inp3 = np.cos(theta)>0  # [0,pi/2],[3pi/2, 2pi]
inp4 = np.cos(theta)<0  # [pi/2, 3pi/2]

# X for boundary conditions
Nxb = 101
# s = np.logspace(0, np.log10(p**((Nxb-1)/2)), int((Nxb-1)/2+1))
# s1 = 0.5*(s-s[0])/(max(s)-min(s))
# xb_den = np.concatenate((s1[:-1], np.flip(1-s1)))[:, None]
xb = np.linspace(0,1,Nxb)[:, None]
X1 = np.hstack((xb, np.ones((Nxb, 1))))      # x [0, 1], y = 1
X2 = np.hstack((xb, np.zeros((Nxb, 1))))     # x [0, 1], y = 0
X3 = np.hstack((np.zeros((Nxb, 1)), xb))     # x = 0, y = [0, 1]
X4 = np.hstack((np.ones((Nxb, 1)), xb))      # x = 1, y = [0, 1]
Xb = np.vstack((X1, X2, X3, X4))

# X test 1D
# the location of the vortex
a = 1
# the test number
Nx = 81
Ny = 81
# x direction
# p = 1.05
# s = np.logspace(0, np.log10(p**((Ny-1)/2)), int((Ny-1)/2+1))
# s1 = 0.5*(s-s[0])/(max(s)-min(s))
# y1 = np.concatenate((s1[:-1], np.flip(1-s1)))
y1 = np.linspace(0,1,Nx)
# y direction
x1 = y1
# meshgrid of xy
[xx, yy] = np.meshgrid(x1, y1)
X_t = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))

# feq = np.sum(np.exp(-mu**2)*mu**2*np.cos(theta)*d_mu*d_theta)

global_var._init()
global_var.set_value('xx', xx)
global_var.set_value('yy', yy)
global_var.set_value('Nx', Nx)
global_var.set_value('Ny', Ny)
global_var.set_value('Nvx', Nvx)
global_var.set_value('Nvy', Nvy)
global_var.set_value('mu', mu)
global_var.set_value('theta', theta)
global_var.set_value('d_mu', d_mu)
global_var.set_value('d_theta', d_theta)
global_var.set_value('X', X)
global_var.set_value('delta', delta)
global_var.set_value('inp1', inp1)
global_var.set_value('inp2', inp2)
global_var.set_value('inp3', inp3)
global_var.set_value('inp4', inp4)
global_var.set_value('Xb', Xb)
global_var.set_value('X1', X1)
global_var.set_value('X2', X2)
global_var.set_value('X3', X3)
global_var.set_value('X4', X4)
global_var.set_value('X_t', X_t)
global_var.set_value('X_train', X)
global_var.set_value('Xb_train', Xb)
global_var.set_value('Nxb', Nxb)

global_var.set_value('c', c)
global_var.set_value('t',t)
global_var.set_value('x',x)
global_var.set_value('y',y)
global_var.set_value('x1', x1)
global_var.set_value('y1', y1)
global_var.set_value('xb', xb)