import random
import sys
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.spatial.distance import cdist, pdist, euclidean

import simulation.bsim as bsim
import behtree.treegen as tg
import behtree.tree_nodes as tree_nodes
import evo.evaluate as evaluate
import evo.operators as op

from matplotlib import animation, rc, rcParams
rcParams['animation.embed_limit'] = 2**128
from IPython.display import HTML



# First set up the figure, the axis, and the plot element we want to animate
fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, figsize=(16,8), dpi=70, facecolor='w', edgecolor='k')
plt.close()

dim = 30
ax1.set_xlim((-dim, dim))
ax1.set_ylim((-dim, dim))

fontsize = 12
ax2.set_xlabel('Time (seconds)', fontsize = fontsize)
ax2.set_ylabel('Percentage of Boxes Collected (%)', fontsize = fontsize)

# Set how data is plotted within animation loop
global line, line1
line, = ax1.plot([], [], 'rh', markersize = 6, markeredgecolor="black", alpha = 0.9)
line1, = ax1.plot([], [], 'bs', markersize = 8, markeredgecolor="black", alpha = 0.5)
box_line, = ax2.plot([],[], 'r-', markersize = 5)

fsize = 12
time_text = ax1.text(-20, 26, '', fontsize = fsize)
box_text = ax1.text(0, 26, '', color = 'red', fontsize = fsize)
line.set_data([], [])
line1.set_data([], [])

def init():
    line.set_data([], [])
    line1.set_data([], [])
    box_line.set_data([], [])
    return (line, line1, box_line,)


# Create the swarm
swarmsize = 50 # Here you can change the size of the swarm.
swarm = bsim.swarm()
swarm.size = swarmsize
swarm.speed = 5
swarm.gen_agents()

# Create the environment
env = bsim.map()
env.env1()
env.gen()
swarm.map = env

# Create the set of boxes to collect
boxes = bsim.boxes()
boxes.set_state('random')
boxes.sequence = False
boxes.radius = 3

# Plot collection reg

# plot the walls
[ax1.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
    [swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], 'k-', lw=2) for a in range(len(swarm.map.obsticles))]

# Set simulation duration
timesteps = 50

ax2.set_xlim((0, timesteps))
ax2.set_ylim((0, 100))
ax2.set_yticks(np.arange(0, 100, 10))
ax2.grid()

# Add agent motion noise
noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))
score = 0

# Here you can change the swarms behvaior
swarm.behaviour = 'rol_anti'
# This value adjusts the rate the agents change their headings
swarm.param = 0.2


field, grid = bsim.potentialField_map(swarm.map)
swarm.field = field
swarm.grid = grid

box_data = []
time_data = []
heat_data = []

def animate(i):

    swarm.iterate(noise[i-1])
    swarm.get_state()
    score = boxes.get_state(swarm, i)

    box_data.append(100*(score/len(boxes.boxes)))
    time_data.append(i)

    time_text.set_text('Time: (%d/%d)' % (i, timesteps))
    
    line1.set_data(boxes.boxes.T[0], boxes.boxes.T[1])
    line.set_data(swarm.agents.T[0], swarm.agents.T[1])
    box_line.set_data(time_data, box_data)

    return (line, line1, box_line, time_text, box_text)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                            frames=timesteps, interval=100, blit=True, cache_frame_data = False)

# Note: below is the part which makes it work on Colab
rc('animation', html='jshtml')
anim
