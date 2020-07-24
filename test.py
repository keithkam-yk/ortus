import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import itertools as it


fig_size = 7
fig = plt.figure(figsize=(fig_size, fig_size))
bounds = [0, 0, 1, 1]
ax = fig.add_axes(bounds, frameon=False)
ax.set_xlim(0, 1), ax.set_xticks([])
ax.set_ylim(0, 1), ax.set_yticks([])

n_cells = 75
d_cells = 0.03

dis1 = 0.01
dis2 = 0.03
dis3 = 0.05
rep_lim = 0.007
adh_lim = 0.0005
v_max = 0.002
heat =  0.0015

A = np.array([[3,    1.5],
              [1.5,  3  ]])

E_A = np.array([[0.2,    0],
                [0,    0.2]])

E_T = np.array([[0.5,    0],
                [0,    0.5]])

W_T = np.array([[1,    1],
                [1,    1]])

W_B = np.array([0.1,   0.1])

T_T = np.array([[5,    -5],
                [-5,    5]])

T_B = np.array([1,    1])

WT_T = np.multiply(W_T,T_T)

D_A = np.array([0.1,    0.1])

D_T = np.array([0.1,    0.1])

H = np.array([0.1,    0.1])

cells = np.zeros(n_cells, dtype=[('position',   float, 2),
                                 ('velocity',   float, 2),
                                 ('edge_color', float, 4),
                                 ('face_color', float, 4),
                                 ('adhesin',    float, 2),
                                 ('promoter',   float, 2),
                                 ('transfac',   float, 2)]
                                 )

cells['position'] = np.random.uniform(0.35, 0.65, (n_cells, 2))

cells['edge_color'][:,3] = np.ones (n_cells)
cells['face_color'][:,3] = np.ones (n_cells) * 0.5

cells['transfac'] = np.random.uniform (0.45,0.55, (n_cells,2))

scat = ax.scatter(cells['position'][:, 0], 
                  cells['position'][:, 1],
                  edgecolors = cells['edge_color'],
                  facecolors = cells['face_color'], 
                  s= (d_cells * fig_size * 72)**2)

def adhesion (dis1, dis2, dis3, rep_lim, adh_lim, dis, cell1, cell2,A ):
    ''' Returns scalar value for force ''' 
    if dis < dis1:
        return -rep_lim
    elif dis < dis2:
        return (-rep_lim + (dis-dis1)*rep_lim/(dis2-dis1))
    else:
        multiplier = np.sum(np.multiply(np.outer(cell1['adhesin'],cell2['adhesin']),A))
        return (adh_lim - abs(dis-(dis2+dis3)/2) * 2*adh_lim/(dis3-dis2)) * multiplier
        

def update(frame_number):
    
    for cell1, cell2 in it.permutations(cells,2):
        displacement = cell2['position'] - cell1['position']
        distance = np.linalg.norm(displacement)
        if distance < dis3:
            direction = displacement/distance
            force = adhesion (dis1, dis2, dis3, rep_lim, adh_lim, distance,cell1,cell2, A)
            cell1['velocity'] += force * direction
            
    cells['velocity'] += np.random.uniform(-heat, heat,(n_cells,2))
    for cell in cells:
        if cell['position'][0] < bounds[0]:
            cell['velocity'][0] *= -1
        if cell['position'][1] < bounds[1]:
            cell['velocity'][1] *= -1
        if cell['position'][0] > bounds[2]:
            cell['velocity'][0] *= -1
        if cell['position'][1] > bounds[3]:
            cell['velocity'][1] *= -1
        velocity = np.linalg.norm(cell['velocity'])
        if velocity > v_max:
            cell['velocity'] *= v_max/velocity
    cells['position'] += cells['velocity']
    
    if frame_number % 10  == 0:
        for cell in cells:
            cell['adhesin']  += np.matmul(E_A, cell['promoter']) - np.multiply(D_A, cell['adhesin'])
            cell['transfac'] += np.matmul(E_T, cell['promoter']) - np.multiply(D_T, cell['transfac'])
            cell['promoter'] =  np.clip((np.matmul(WT_T, cell['transfac']) + T_B)/(np.matmul(W_T, cell['transfac']) + W_B) * H , 0,1)
            
    cells['face_color'][:,0] = np.clip(cells['adhesin'][:,0],0,1)
    cells['face_color'][:,1] = np.clip(cells['adhesin'][:,1],0,1)
    
    scat.set_offsets(cells['position'])
    scat.set_edgecolors(cells['edge_color'])
    scat.set_facecolors(cells['face_color'])
    print(frame_number)
    
animation = FuncAnimation(fig, update, interval = 20, blit = False)
plt.show()
