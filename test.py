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

K = np.array([[3,    1.5],
              [1.5,  1  ]])

cells = np.zeros(n_cells, dtype=[('position',   float, 2),
                                 ('velocity',   float, 2),
                                 ('edge_color', float, 4),
                                 ('face_color', float, 4),
                                 ('adhesin',    float, 2)]
                                 )

cells['position'] = np.random.uniform(0.35, 0.65, (n_cells, 2))

cells['edge_color'][:,3] = np.ones (n_cells)

#cells['face_color'] = np.ones ((n_cells, 4))
#cells['face_color'][:,3] = np.ones (n_cells) * 0.5

scat = ax.scatter(cells['position'][:, 0], 
                  cells['position'][:, 1],
                  edgecolors = cells['edge_color'],
                  facecolors = cells['face_color'], 
                  s= (d_cells * fig_size * 72)**2)

for i, cell in enumerate (cells):
    if i % 2 == 1:
        cell['face_color'] = [1,0,0,0.5]
        cell['adhesin'] = [1,0]
    else:
        cell['face_color'] = [0,1,0,0.5]
        cell['adhesin'] = [0,1]
            

def adhesion (dis1, dis2, dis3, rep_lim, adh_lim, dis, cell1, cell2, K ):
    ''' Returns scalar value for force ''' 
    if dis < dis1:
        return -rep_lim
    elif dis < dis2:
        return (-rep_lim + (dis-dis1)*rep_lim/(dis2-dis1))
    else:
        multiplier = np.sum(np.multiply(np.outer(cell1['adhesin'],cell2['adhesin']),K))
        return (adh_lim - abs(dis-(dis2+dis3)/2) * 2*adh_lim/(dis3-dis2)) * multiplier
        

def update(frame_number):
    
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
            
    for cell1, cell2 in it.permutations(cells,2):
        displacement = cell2['position'] - cell1['position']
        distance = np.linalg.norm(displacement)
        if distance < dis3:
            direction = displacement/distance
            force = adhesion (dis1, dis2, dis3, rep_lim, adh_lim, distance,cell1,cell2, K)
            cell1['velocity'] += force * direction
    cells['position'] += cells['velocity']       
    
    for cell in cells:
        velocity = np.linalg.norm(cell['velocity'])
        if velocity > v_max:
            cell['velocity'] *= v_max/velocity
    
    scat.set_offsets(cells['position'])
    scat.set_edgecolors(cells['edge_color'])
    scat.set_facecolors(cells['face_color'])
    
    
animation = FuncAnimation(fig, update, interval =1, blit = False)
plt.show()
