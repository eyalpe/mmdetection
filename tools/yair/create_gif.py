import glob
import imageio

filenames = glob.glob('/home/yairshe/results/mmdetection_hackathon/task_1_visualize_objectness/*.png')
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('/home/yairshe/results/mmdetection_hackathon/task_1_visualize_objectness/movie.gif', images, duration=2)