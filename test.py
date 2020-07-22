import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, 1), ax.set_xticks([])
ax.set_ylim(0, 1), ax.set_yticks([])

n_cells = 10
cells = np.zeros(n_cells, dtype=[('position', float, 2),
                                 ('velocity', float, 2)]
                                 )

cells['position'] = np.random.uniform(0, 1, (n_cells, 2))
cells['velocity'] = np.random.uniform(-0.001, 0.001, (n_cells, 2))

scat = ax.scatter(cells['position'][:, 0], cells['position'][:, 1])
# s=cells['size'], lw=0.5, facecolors='none'

def update(frame_number):
    cells['position'] += cells['velocity']
    scat.set_offsets(cells['position'])
    
animation = FuncAnimation(fig, update, interval=10)
plt.show()
