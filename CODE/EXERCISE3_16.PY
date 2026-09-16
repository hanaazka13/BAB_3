# Exercise 3.16 - Modified Q_test.py for 8-State Routing
import numpy as np
import matplotlib.pyplot as plt
from Q_Utils import *

# Setting up Parameters =========================================
points_list = [
    (0, 1), (1, 2), (1, 3), 
    (2, 4), (3, 5), (4, 6), 
    (5, 7), (6, 7)
]

# Set goal ke state 7
goal = 7

# Show the routing graph
showgraph(points_list)

# Ubah ukuran matriks menjadi 8
MATRIX_SIZE = 8

# Create matrix R & Q
R = createRmat(MATRIX_SIZE, points_list, goal)
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# Learning parameter
gamma = 0.8

# Training ======================================================
scores = []
for i in range(1000):  # Naikkan episode sedikit agar 8 state ter-explore sempurna
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(R, current_state)
    action = sample_next_action(available_act)
    score = update(R, Q, current_state, action, gamma)
    scores.append(score)

print("Trained Q matrix:")
print(Q / np.max(Q) * 100)

# Testing =======================================================
current_state = 0
steps = [current_state]

while current_state != goal:
    q_row = Q[current_state, :]
    max_val = np.max(q_row)
    max_indices = np.where(np.array(q_row).ravel() == max_val)[0]
    
    if len(max_indices) > 1:
        next_step_index = int(np.random.choice(max_indices))
    else:
        next_step_index = int(max_indices[0])

    steps.append(next_step_index)
    current_state = next_step_index

# Display Results ===============================================
print("Most efficient path:")
print(steps)

plt.plot(scores)
plt.show()