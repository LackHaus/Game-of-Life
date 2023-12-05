import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2

if "Frames" not in os.listdir():
    os.mkdir("Frames")

if "Frames" in os.listdir():
    for file in os.listdir("Frames/"):
        os.remove("Frames/" + file)

fill_factor = 0.05
epochs = 12
gridSize = (50,50)

grid = np.zeros(gridSize)
grids = [grid]

for i in range(gridSize[0]):
    for j in range(gridSize[1]):
        if random.random() > fill_factor:
            grid[i,j] = 1


image = plt.imshow(grid)
plt.title("Initial State (Close to start 'Life')")
plt.show()


for p in range(epochs):
    print("Epoch:" + str(p))
    for i in range(gridSize[0]):
        for j in range(gridSize[1]):
            neighbors = 0
            neighbors += grid[np.mod(i-1, gridSize[0]),np.mod(j-1, gridSize[1])]
            neighbors += grid[np.mod(i-1, gridSize[0]),np.mod(j, gridSize[1])]
            neighbors += grid[np.mod(i-1, gridSize[0]),np.mod(j+1, gridSize[1])]
            neighbors += grid[np.mod(i, gridSize[0]),np.mod(j-1, gridSize[1])]
            neighbors += grid[np.mod(i, gridSize[0]),np.mod(j+1, gridSize[1])]
            neighbors += grid[np.mod(i+1, gridSize[0]),np.mod(j-1, gridSize[1])]
            neighbors += grid[np.mod(i+1, gridSize[0]),np.mod(j, gridSize[1])]
            neighbors += grid[np.mod(i+1, gridSize[0]),np.mod(j+1, gridSize[1])]
            if grid[i,j] == 1:
                if neighbors < 2 or neighbors > 3:
                    grid[i,j] = 0
            else:
                if neighbors == 3:
                    grid[i,j] = 1
    grids.append(grid)


    plt.imshow(grid)
    plt.savefig('Frames/'+str(p)+'.png')

print("Producing movie")

image_folder = 'Frames'
video_name = 'life.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 5, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
