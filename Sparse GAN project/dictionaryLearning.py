import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist,mnist
from sklearn.decomposition import DictionaryLearning
import pickle
from os import path
from random import randint
from numpy import savetxt
img_rows, img_cols, channels = 28, 28, 1
sparse_dim = img_rows * img_cols * channels
THR = 1
use_fashion = False


def thr(x, Lambda):
  x[np.abs(x)<Lambda] = 0
  return x

if use_fashion:
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
else:
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # normalize between +1 -1

x_train = x_train.reshape(-1, img_rows*img_cols*channels) # each image as vector

np.random.shuffle(x_train)
print(x_train.shape)

#dictionary file name
if use_fashion:
  file_name = 'dictionary_fashion_mnist_undercomplete'
else:
  file_name = 'dictionary_mnist_overcomplete'
#check if dictionary exists
if not path.exists(file_name):
  d=DictionaryLearning(n_components=2*784, max_iter=20)
  # train dictionary
  d.fit(x_train[1:10000, :])
  dictionary = d.components_
  print(dictionary.shape)

  with open(file_name, 'wb') as output:
    pickle.dump(d, output, pickle.HIGHEST_PROTOCOL)
  print("created new dictionary")

else:
  with open(file_name, 'rb') as input:
    d = pickle.load(input)
  print("loaded dictionary")
  sparse_dict = np.transpose(d.components_)
  print("analyse pursuit")

  num_images_to_pursuit = 10
  mean_support=0
  average_element_size = 0
  all_support_coeffs = np.array([])
  for i in range(num_images_to_pursuit):

    idx = randint(0, x_train.shape[0])
    sparse_vec = d.transform(x_train[idx:idx+1,:])
    all_support_coeffs = np.append(all_support_coeffs, sparse_vec[sparse_vec!=0])
    mean_support += np.count_nonzero(sparse_vec)
    average_element_size += np.average(np.abs(sparse_vec[sparse_vec!=0]))
  print("mean support is "+ str(mean_support/num_images_to_pursuit))
  print("average_atom_coeff is " + str(average_element_size / num_images_to_pursuit))
  #plt.hist(all_support_coeffs,bins=100)
  #plt.show()
  thrs = [0,0.01,0.1,0.5,1,2]
  figs, axs = plt.subplots(num_images_to_pursuit, len(thrs)+1)
  axs[0][0].set_title('bla')

  for k in range(1,len(thrs)+1):
    axs[0][k].set_title("thr "+str(thrs[k-1]))

  for i in range(num_images_to_pursuit):
    idx = randint(0, x_train.shape[0])
    im = x_train[idx, :]
    im = np.array(im)
    axs[i,0].imshow(im.reshape(28,28), cmap="gray")
    axs[i,0].axis('off')
    sparse_vec = d.transform(x_train[idx:idx + 1, :])
    for j in range(len(thrs)):
      sparse_vec_thr = thr(sparse_vec, thrs[j])
      im_r = np.matmul(sparse_dict, np.transpose(sparse_vec_thr))
      axs[i,j+1].imshow(im_r.reshape(28,28), cmap="gray")
      axs[i, j+1].axis('off')
  plt.axis('off')
  plt.show()

  transformed_x = d.transform(x_train)
  savetxt('transformed_x.csv', transformed_x, delimiter=',')