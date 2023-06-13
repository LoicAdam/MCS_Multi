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

nb1 = 4
nb2 = 3
nb = nb1 + nb2
a1 = 1.5
a2 = -1.5
b1 = 0
b2 = 1.5

np.random.seed(28) #Original seed: 28.

x1 = np.asarray([0.1,0.2,0.3,0.45])
x2 = np.asarray([0.65,0.8,0.9])
x = np.hstack((x1,x2))

y1 = generate_points_linear([a1], [b1], x1, np.zeros(x1.shape[0]))
y2 = generate_points_linear([a2], [b2], x2, np.zeros(x2.shape[0]))
y = np.hstack((y1,y2))

boxes = generate_hyperrectangles(x, y, x_err_bounds = (0.01,0.015), y_err_bounds = (0.02,0.04)).T

if x.ndim == 1:
    x = x[:,None]
    x1 = x1[:,None]
    x2 = x2[:,None]
    
reg_outliner = LinearRegression().fit(x, y)
reg_truth_1 = LinearRegression().fit(x1,y1)
reg_truth_2 = LinearRegression().fit(x2,y2)

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
x_ori_1 = np.linspace(0,0.5,100)
x_ori_2 = np.linspace(0.5,1,100)

y_lin_outliner = reg_outliner.coef_ * x_ori + reg_outliner.intercept_
y_lin_truth_1 = reg_truth_1.coef_ * x_ori_1 + reg_truth_1.intercept_
y_lin_truth_2 = reg_truth_2.coef_ * x_ori_2 + reg_truth_2.intercept_

ax.plot(x_ori, y_lin_outliner, linestyle = '--')
ax.plot(x_ori_1, y_lin_truth_1)
ax.plot(x_ori_2, y_lin_truth_2)
#ax.scatter(x,y,s=10)

ax.set_ylim(0,1)
ax.set_xlim(0,1)
label = ax.set_xlabel('x', fontsize = 9)
ax.xaxis.set_label_coords(1.025, +0.025)
label = ax.set_ylabel('y', fontsize = 9, rotation=0)
ax.yaxis.set_label_coords(0,1.025)

fig.set_dpi(300.0)
tikzplotlib.clean_figure()
tikzplotlib.save('results/multi_linear_regression.tex')
plt.savefig('results/multi_linear_regression.png', dpi=300)

###

combination_boxes = list(itertools.combinations(range(0,nb), 2))
fig, ax = plt.subplots()
ax.set_ylim(-1,3)
ax.set_xlim(-3,3)

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
    
ax.annotate(text = '*', xy = (a1, b1), color = 'red', va='center', ha='center', size = 8)
ax.annotate(text = '*', xy = (a2, b2), color = 'red', va='center', ha='center', size = 8)
ax.annotate(text = '$(a_1^*,b_1^*)$', xy =  (a1, b1-0.45), va='center', ha='center', color = 'red', size = 8)
ax.annotate(text = '$(a_2^*,b_2^*)$', xy =  (a2, b2+0.15), va='center', ha='center', color = 'red', size = 8)

ax.annotate(text = '$c_1 = \{(3, 4), (3, 5), (3, 6),$ \n $(4, 5), (4, 6), (5, 6)\}$', 
            xy = (0.25, 1.5), va='center', ha='center', size = 8)
ax.annotate(text = '$c_2 = \{(0, 1), (0, 2), (0, 3),$ \n $ (1, 2), (1, 3), (2, 3)\}$', 
            xy = (1.75, 0.5), va='center', ha='center', size = 8)


label = ax.set_xlabel('a', fontsize = 9)
ax.xaxis.set_label_coords(1.025, +0.025)
label = ax.set_ylabel('b', fontsize = 9, rotation=0)
ax.yaxis.set_label_coords(0,1.025)
    
fig.set_dpi(300.0)
tikzplotlib.clean_figure()
tikzplotlib.save('results/boxes_multi_linear.tex')
plt.savefig('results/boxes_multi_linear.png', dpi=300)

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