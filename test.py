import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import time
import random

from matplotlib.animation import FuncAnimation
from scipy.spatial import distance_matrix
fig_size = 7
fig = plt.figure(figsize=(fig_size, fig_size))
bounds = [0, 0, 1, 1]
ax = fig.add_axes(bounds, frameon=False)
ax.set_xlim(0, 1), ax.set_xticks([])
ax.set_ylim(0, 1), ax.set_yticks([])

n_cells = 25
d_cells = 0.03

dis1 = 0.01
dis2 = 0.03
dis3 = 0.05
rep_lim = 0.01
adh_lim = 0.0005
v_max = 0.002
heat =  0.0015      

A = np.array([[3,    1.5],      # Adhesin interactions
              [1.5,  3  ]])

E_A = np.array([[0.2,    0],    # Expression of adhesin
                [0,    0.2]])

E_T = np.array([[0.5,    0],    # Expression of transcription factor
                [0,    0.5]])

E_S = np.array([[0.5,    0],    # Expression of surface signal 
                [0,    0.5]])

E_R = np.array([[0.5,    0],    # Expression of receptor
                [0,    0.5]])

E_B = np.array([0.5,    0.5])    # Expression of receptor

W_T = np.array([[1,    1],      # Weighting of transcription factor activation
                [1,    1]])

W_B = np.array([0.1,   0.1])    # Weighting of basal activation

W_S = np.array([[0,    1],      # Weighting of surface signal to receptor 
                [1,    0]])

T_T = np.array([[5,    -5],     # Target activation level for transcription factors
                [-5,    5]])

T_B = np.array([1,    1])       # Target activation level for basal actication

T_R = np.array([[0,    0],     # Target activation level for receptors
                [0,    0]])

D_A = np.array([0.1,    0.1])   # Decay rate for adhesin

D_T = np.array([0.1,    0.1])   # Decay rate for transcription factor

D_R = np.array([0.1,    0.1])   # Decay rate for receptor

D_S = np.array([0.1,    0.1])   # Decay rate for surface signal

D_B = np.array([0.1])           # Decay rate for birth signal

H = np.array([0.1,    0.1])     # Hill coeff nonlinearity for promoter activation

WT_T = np.multiply(W_T,T_T)     # Precalculation for transcription factor activation weighting

cells = np.zeros(n_cells, dtype=[('position',       float, 2),
                                 ('velocity',       float, 2),
                                 ('edge_color',     float, 4),
                                 ('face_color',     float, 4),
                                 ('adhesin',        float, 2),
                                 ('promoter',       float, 2),
                                 ('trans_factor',   float, 2),
                                 ('receptor',       float, 2),
                                 ('surface_out',    float, 2),
                                 ('surface_in',     float, 2),
                                 ('receptor_act',   float, 2),
                                 ('birth',          float, 1),
                                 ('alive',          bool,  1)
                                 ])

cells[0]['alive'] = True
cells['position'] = np.random.uniform(0.45, 0.55, (n_cells, 2))
cells['edge_color'][:,3] = np.ones (n_cells)
cells['face_color'][:,3] = np.ones (n_cells) * 0.5
cells['trans_factor'] = np.random.uniform (0.45,0.55, (n_cells,2))
multipliers = np.zeros((n_cells,n_cells))

scat = ax.scatter(cells['position'][:, 0], 
                  cells['position'][:, 1],  
                  edgecolors = cells['edge_color'],
                  facecolors = cells['face_color'], 
                  s= (d_cells * fig_size * 72)**2)

def adhesion (dis1, dis2, dis3, rep_lim, adh_lim, dis, multiplier):
    ''' Returns scalar value for force ''' 
    if dis < dis1:
        return -rep_lim
    elif dis < dis2:
        return (-rep_lim + (dis-dis1)*rep_lim/(dis2-dis1))
    else:
        return (adh_lim - abs(dis-(dis2+dis3)/2) * 2*adh_lim/(dis3-dis2)) * multiplier
        

def update(frame_number):
    global cells
    
    if frame_number % 10  == 0:
        cells['surface_in'] = np.zeros ((n_cells,2))
        
            
    distances =  distance_matrix(cells['position'],cells['position'])
    
    for i1, cell1 in enumerate(cells):
        for i2, cell2 in enumerate(cells):
            if i1 == i2 or cell1['alive'] == False or cell2['alive'] == False:
                continue
            distance = distances[i1,i2]
            if distance < dis3:
                if frame_number % 10  == 0:
                    cell1['surface_in'] += cell2['surface_out']*np.exp(-50*distance)
                    multipliers[i1,i2] = np.sum(np.multiply(np.outer(cell1['adhesin'],cell2['adhesin']),A))
                displacement = cell2['position'] - cell1['position']
                direction = displacement/distance
                force = adhesion (dis1, dis2, dis3, rep_lim, adh_lim, distance, multipliers[i1,i2])
                cell1['velocity'] += force * direction
                
    cells['velocity'] += np.random.uniform(-heat, heat,(n_cells,2))
    
    
    if frame_number % 10  == 0:
        tot_cells = sum(cells['alive'])
        new_cells = []
        for i, cell in enumerate(cells):
            if cell['alive'] == False:
                continue
            cell['adhesin']         += np.matmul(E_A, cell['promoter']) - np.multiply(D_A, cell['adhesin'])
            cell['trans_factor']    += np.matmul(E_T, cell['promoter']) - np.multiply(D_T, cell['trans_factor'])
            cell['surface_out']     += np.matmul(E_S, cell['promoter']) - np.multiply(D_S, cell['surface_out'])
            cell['receptor']        += np.matmul(E_R, cell['promoter']) - np.multiply(D_R, cell['receptor'])
            cell['birth']           += np.matmul(E_B, cell['promoter']) - np.multiply(D_B, cell['birth'])            
            cell['promoter']        =  np.clip((np.matmul(WT_T, cell['trans_factor']) + T_B)/(np.matmul(W_T, cell['trans_factor']) + W_B) * H , 0,1)
            cell['receptor_act']    = np.multiply(np.matmul(W_S,cell['surface_in']), cell['receptor'])
            if tot_cells < n_cells:
                if random.random()*10 < np.amax(cell['birth']):
                    new_cells.append(i)
        if tot_cells < n_cells:
            for j in new_cells:
                for k, try_cell in enumerate (cells):
                    if try_cell['alive'] == False:
                        cells[k] = cells[new_cells[j]]
                        cells[k]['position'] += np.random.uniform(-0.03,0.03,2)
                        break
    
    velocities = np.linalg.norm(cells['velocity'],axis = 1)
    for i, cell in enumerate(cells):
        if cell['alive'] == False:
            continue
        if cell['position'][0] < bounds[0]:
            cell['velocity'][0] *= -1
        if cell['position'][1] < bounds[1]:
            cell['velocity'][1] *= -1
        if cell['position'][0] > bounds[2]:
            cell['velocity'][0] *= -1
        if cell['position'][1] > bounds[3]:
            cell['velocity'][1] *= -1
        if velocities[i] > v_max:
            cell['velocity'] *= v_max/velocities[i]

    cells['position'] += cells['velocity']
    
    
    cells['face_color'][:,0] = np.clip(cells['adhesin'][:,0],0,1)
    cells['face_color'][:,1] = np.clip(cells['adhesin'][:,1],0,1)
    cells['face_color'][:,2] = np.clip(cells['receptor_act'][:,0],0,1)
    
    scat.set_offsets(cells['position'][cells['alive']==True])
    scat.set_edgecolors(cells['edge_color'][cells['alive']==True])
    scat.set_facecolors(cells['face_color'][cells['alive']==True])
    #print(frame_number)

animation = FuncAnimation(fig, update, frames = 1000,interval = 20, blit = False)
plt.show()

#%%
start_time = time.time()
for i in range (1000):
    update(i)
print("--- %s seconds ---" % (time.time() - start_time))
#%%
import cProfile
cProfile.run('update(10)',sort='cumulative')
#%%
cProfile.run('update(1)',sort='cumulative')
