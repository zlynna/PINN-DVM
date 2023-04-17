import module as global_var
import numpy as np
import pandas as pd
from pyDOE import *

### rarefied parameter ###
delta = 1
########### data generation ############
### spatial grid ###
# configuration parameters
Bond = 1.
A = Bond**2/2
T = Bond*2+Bond*np.sqrt(2)
Dh = 4*A/T
# channel part
N_train = 5000
r1 = np.sqrt(np.random.random((N_train,1)))
r2 = np.random.random((N_train,1))
x_train = r1*Bond-r1*r2*Bond
y_train = r1*r2*Bond
X_train = np.hstack((x_train, y_train))/Dh

### velocity grid ###
Nvx = 8
Nvy = 50
v_max = 7
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
d_theta = np.ones_like(d_theta)*2*np.pi/Nvy

# theta for boundary conditions
top = np.sin(theta)<0     # [pi, 2*pi]
bottom = np.sin(theta)>0  # [0, pi]
left = np.cos(theta)>0    # [0,pi/2],[3pi/2, 2pi]
right = np.cos(theta)<0   # [pi/2, 3pi/2]

edge = theta<7/4*np.pi   
edge = 3/4*np.pi<edge*theta

# X for boundary conditions
Nxb = 250
Xb_left = np.hstack((np.zeros((Nxb, 1)), np.linspace(0,1,Nxb)[:, None]))*Bond/Dh
Xb_bot = np.hstack((np.linspace(0,1,Nxb)[:, None], np.zeros((Nxb, 1))))*Bond/Dh
Xb_edge = np.hstack((np.linspace(0,1,Nxb)[:, None], 1-np.linspace(0,1,Nxb)[:, None]))*Bond/Dh
X_b = {'Xb_bot': Xb_bot, 
       'Xb_left': Xb_left, 
       'Xb_edge': Xb_edge}

### test data ###
# the test number
# N_test = 10000
# r1 = np.sqrt(np.random.random((N_test,1)))
# r2 = np.random.random((N_test,1))
# x_test = r1*Bond-r1*r2*Bond
# y_test = r1*r2*Bond
# X_test = np.hstack((x_test, y_test))/Dh

x_test, y_test = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
X_test = np.hstack((x_test.ravel()[:,None], y_test.ravel()[:,None]))/Dh

# symmetry
x_test = np.linspace(0,0.5, 100)[:, None]
y_test = x_test
X_test1 = np.hstack((x_test, y_test))/Dh

### data transport ###
global_var._init()
global_var.set_value('Nvx', Nvx)
global_var.set_value('Nvy', Nvy)
global_var.set_value('mu', mu)
global_var.set_value('theta', theta)
global_var.set_value('c', c)
global_var.set_value('t', t)
global_var.set_value('d_mu', d_mu)
global_var.set_value('d_theta', d_theta)
global_var.set_value('X_train', X_train)
global_var.set_value('delta', delta)
global_var.set_value('top', top)
global_var.set_value('bottom', bottom)
global_var.set_value('left', left)
global_var.set_value('right', right)
global_var.set_value('edge', edge)
global_var.set_value('X_test', X_test)
global_var.set_value('Nxb', Nxb)
global_var.set_value('X_b', X_b)