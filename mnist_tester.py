from layer import Layer
import mnist
import numpy as np
import random

images = mnist.train_images()
labels = mnist.train_labels()

feature_length = images.shape[1] * images.shape[2]
image_count = images.shape[0]

images = images.reshape((image_count, feature_length))
images = images / 255
images = images - np.mean(images)

L1 = Layer(size=feature_length, inputs=feature_length)
L2 = Layer(size=int(feature_length/10), inputs=feature_length)
L3 = Layer(size=10, inputs=int(feature_length/10))

L1.next = L2
L2.next = L3

print('Training Process')
for i in range (5000):
    index = random.randint(0, image_count)
    elist = [[0] for j in range(10)]
    elist[labels[index]][0] = 0.8
    expected = np.asarray(elist)
    sample = images[index].reshape(feature_length,1)
    L1.back_propagate(sample, expected)

print('Testing')
for i in range(10):
    index = random.randint(0, image_count)
    elist = [[0] for j in range(10)]
    elist[labels[index]][0] = 0.8
    expected = np.asarray([elist])
    sample = images[index].reshape(feature_length,1)
    print('Target: ', labels[index])
    output = L1.apply_chain(sample)
    print('Result: ', np.argmax(output))
    print(output.transpose())