
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# Set random seed for reproducibility
np.random.seed(67)

# Simulation window parameters
xMin = 0
xMax = 1
yMin = 0
yMax = 1
xDelta = xMax - xMin
yDelta = yMax - yMin
areaTotal = xDelta * yDelta

# Point process parameters
lambda0 = 10

# Simulate a Poisson point process
numbPoints = np.random.poisson(lambda0 * areaTotal)
xx = xDelta * np.random.uniform(0, 1, numbPoints) + xMin
yy = yDelta * np.random.uniform(0, 1, numbPoints) + yMin
print("Number of clients: ",len(xx))

# # Plotting
# fig, ax = plt.subplots()
# for x, y in zip(xx, yy):
#     color = np.random.rand(3,)  # Generate a random color for each point
#     circle = plt.Circle((x, y), 0.3, color=color, fill=False)
#     ax.add_patch(circle)
#     ax.scatter(x, y, color=color)

rng = 1.005
# Plotting
fig, ax = plt.subplots()
total = 0
for i, (x, y) in enumerate(zip(xx, yy)):
    color = np.random.rand(3,)  # Generate a random color for each point
    # circle = plt.Circle((x, y), 0.1, color=color, fill=False)
    # ax.add_patch(circle)
    ax.scatter(x, y, color=color)

    # Count points within a distance of 0.3
    count = 0
    for j in range(len(xx)):
        if i != j and np.sqrt((xx[i] - xx[j])**2 + (yy[i] - yy[j])**2) <= rng:
            count += 1

    print(count)
    total = total + count
print("Average: ", total/10)

plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')  # Set aspect ratio to be equal
plt.show()