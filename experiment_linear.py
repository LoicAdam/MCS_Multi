# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt 
import itertools
import tikzplotlib
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression

from model.algorithm_mcs import mcs_from_intervals
from model.linear_function import points_to_generate, generate_points_linear, generate_hyperrectangles
from model.list_inter import intersection_sources

nb = 5
a = 0.5
b = 0.2

np.random.seed(28) #Original seed: 28.

#x = points_to_generate(nb,1,0,1)
x = np.asarray([0.1,0.3,0.5,0.7,0.9])
y = generate_points_linear([a], [b], x, [0,0,0.3,0,0])
boxes = generate_hyperrectangles(x, y, x_err_bounds = (0.01,0.02), y_err_bounds = (0.02,0.05)).T

if x.ndim == 1:
    x = x[:,None]
    
reg_outliner = LinearRegression().fit(x, y)
reg_truth = LinearRegression().fit(np.delete(x, 2)[:,np.newaxis], np.delete(y, 2))

fig, ax = plt.subplots()
for i in range(0, nb):
    
    ax.add_patch(Rectangle((boxes[0,i,0], boxes[0,i,1]), 
                           width = boxes[1,i,0]-boxes[0,i,0], 
                           height = boxes[1,i,1]-boxes[0,i,1],
                           alpha = 0.2, label="_nolegend_"))
    ax.annotate(i, (boxes[0,i,0] + (boxes[1,i,0]-boxes[0,i,0])/2,
                    boxes[0,i,1] + (boxes[1,i,1]-boxes[0,i,1])/2 -0.1),
                ha='center')

x_ori = np.linspace(0,2,100)
y_lin_outliner = reg_outliner.coef_ * x_ori + reg_outliner.intercept_
y_lin_truth = reg_truth.coef_ * x_ori + reg_truth.intercept_
ax.plot(x_ori, y_lin_outliner, linestyle = '--')
ax.plot(x_ori, y_lin_truth)
#ax.scatter(x,y,s=10)

ax.set_ylim(0,1)
ax.set_xlim(0,1)
label = ax.set_xlabel('x', fontsize = 9)
ax.xaxis.set_label_coords(1.025, +0.025)
label = ax.set_ylabel('y', fontsize = 9, rotation=0)
ax.yaxis.set_label_coords(0,1.025)

fig.set_dpi(300.0)
tikzplotlib.clean_figure()
tikzplotlib.save('results/linear_regression.tex')
plt.savefig('results/linear_regression.png', dpi=300)

###

combination_boxes = list(itertools.combinations(range(0,nb), 2))
fig, ax = plt.subplots()
ax.set_ylim(-1,2)
ax.set_xlim(-2,3)

intervals_dim_1 = np.zeros((len(combination_boxes),2))
intervals_dim_2 = np.zeros((len(combination_boxes),2))
   
i = 0
for couple_box in combination_boxes:
    
    ind_box_1 = couple_box[0]
    ind_box_2 = couple_box[1]
    
    box_1 = boxes[:,ind_box_1,:]
    box_2 = boxes[:,ind_box_2,:]
    
    m_list = np.zeros(4)
    m_list[0] = ((box_1[1,1]) - (box_2[0,1])) / (box_1[0,0] - (box_2[1,0]))
    m_list[1] = ((box_1[1,1]) - (box_2[0,1])) / ((box_1[1,0]) - box_2[0,0])
    m_list[2] = ((box_1[0,1]) - (box_2[1,1])) / ((box_1[1,0]) - box_2[0,0])
    m_list[3] = ((box_1[0,1]) - (box_2[1,1])) / (box_1[0,0] - (box_2[1,0]))
    
    m_min = np.min(m_list)
    m_max = np.max(m_list)
    
    p_list = np.zeros(4)
    p_list[0] = (box_1[1,1]) - m_list[0] * box_1[0,0]
    p_list[1] = (box_1[1,1]) - m_list[1] * (box_1[1,0])
    p_list[2] = box_1[0,1] - m_list[2] * (box_1[1,0])
    p_list[3] = box_1[0,1] - m_list[3] * box_1[0,0]

    p_min = np.min(p_list)
    p_max = np.max(p_list)
    
    intervals_dim_1[i,:] = [m_min,m_max]
    intervals_dim_2[i,:] = [p_min,p_max]
    
    ax.add_patch(Rectangle(xy = (m_min, p_min), 
                           width = m_max-m_min, 
                           height = p_max-p_min,
                           ec = (0,0,0,0.8),
                           fc = (0,0.5,1,0.15),
                           label="_nolegend_"))
    
    #ax.annotate(text = i,
    #            xy = (m_min + (m_max-m_min)/2, p_min + (p_max-p_min)/2))
    print(m_min, m_max, p_min, p_max)
    
    i = i+1
    
ax.annotate(text = '*', xy = (a, b), color = 'red', va='center', ha='center', size = 8)
ax.annotate(text = '$(a^*,b^*)$', xy =  (a, b-0.2), va='center', ha='center', color = 'red', size = 8)

ax.annotate(text = '$c_1 = \{(2,3)\}$', xy = (-1.05, 1.3), va='center', ha='center', size = 8)
ax.annotate(text = '$c_2 = \{(2,4)\}$', xy = (0.4, 0.9), va='center', ha='center', size = 8)
ax.annotate(text = '$c_3 = \{(0, 1), (0, 3), (0, 4),$ \n $(1, 3), (1, 4), (3, 4)\}$', xy = (-0.75, 0.15), va='center', ha='center', size = 8)
ax.annotate(text = '$c_4 = \{(0,2),(3,4)\}$', xy = (1.65, 0.35), va='center', ha='center', size = 8)
ax.annotate(text = '$c_5 = \{(1,2)\}$', xy =  (2.15, -0.3), va='center', ha='center', size = 8)

label = ax.set_xlabel('a', fontsize = 9)
ax.xaxis.set_label_coords(1.025, +0.025)
label = ax.set_ylabel('b', fontsize = 9, rotation=0)
ax.yaxis.set_label_coords(0,1.025)
    
fig.set_dpi(300.0)
tikzplotlib.clean_figure()
tikzplotlib.save('results/boxes_linear.tex')
plt.savefig('results/boxes_linear.png', dpi=300)

test_final = []

test_1 = mcs_from_intervals(intervals_dim_1)
test_2 = mcs_from_intervals(intervals_dim_2)

test_final.append(test_1)
test_final.append(test_2)

hur = intersection_sources(test_final)
R = np.zeros((len(hur),nb,nb)) 

t = 0
for i in hur:
    for j in i:
        print(combination_boxes[j], end = ' ')
        R[t, combination_boxes[j][0], combination_boxes[j][1]] = 1
    print('')
    t = t+1
plt.show()


j = 0
for i in combination_boxes:
    print(j, end = ':')
    print(combination_boxes[j])
    j = j+1