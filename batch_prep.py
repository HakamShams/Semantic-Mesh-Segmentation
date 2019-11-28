'''

Utility mesh function for batch generation

Author: Hakam Shams
Date: Novemebr 2019

Input:  root       : data path
        num_faces  : number of sampled faces, default 8000
        nb_classes : number of classes, default 8
        scale      : scale to unite sphere for PointNet, default False
        sampling   : sampling method [random, fps, or knn], default random
        mode       : train or val, default train

Output: Class HessigheimDataset, get items: data	numpy array NxF
											label	numpy array Nx1
											weight	numpy array Nx1

Dependencies: numpy - os - h5py - open3d - scipy - sklearn - matplotlib

'''

import numpy as np
import os
import h5py
import open3d
from scipy.spatial import cKDTree
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HessigheimDataset():

	def __init__(self, root, num_faces=8000, nb_classes=8, scale=False, sampling='random', mode='train'):
		self.root = root
		self.num_faces = num_faces
		self.mode = mode
		self.sampling = sampling
		self.nb_classes = nb_classes
		self.scale = scale

		files = os.listdir(self.root)

		self.data_all = None
		self.label_all = None

		for file in files:

			if file.endswith('h5'):
				hdf = h5py.File(self.root + '/{}'.format(file), mode='r')
				face_tile = np.array(hdf['data'])
				label_tile = np.array(hdf['label'])

			elif file.endswith('txt'):
				data = np.loadtxt(self.root + '/{}'.format(file))
				face_tile = data[:,:-1]
				label_tile = data[:,-1]

			else:
				continue

			if face_tile.shape[0] < self.num_faces:
				continue

			if self.sampling == 'random':

				indices = np.random.choice(np.array(face_tile).shape[0], self.num_faces, replace=False)

				face_cur = np.take(face_tile, indices, 0)
				label_cur = np.take(label_tile, indices)

			elif self.sampling == 'knn':

				tree = cKDTree(face_tile[:,0:3])

				center = [face_tile[:,0].mean(), face_tile[:,1].mean(), face_tile[:,2].mean()]

				_, ind = tree.query(center, k=self.num_faces)

				face_cur = face_tile[ind]
				label_cur = label_tile[ind]

			elif self.sampling == 'fps':

				data_cur = np.concatenate((face_tile, np.expand_dims(label_tile, -1)), axis=1)

				data_cur = self.graipher(data_cur, self.num_faces)
				face_cur = data_cur[:, :-1]
				label_cur = data_cur[:, -1]

			face_cur = np.expand_dims(face_cur, axis=0)
			label_cur = np.expand_dims(label_cur, axis=0)

			if self.data_all is None:
				self.data_all = face_cur
				self.label_all = label_cur
			else:
				self.data_all = np.concatenate((self.data_all, face_cur), axis=0)
				self.label_all = np.concatenate((self.label_all, label_cur))

		if self.mode == 'train':
			weights = np.zeros(self.nb_classes)

			for sem in self.label_all:
				tmp, _ = np.histogram(sem, range(self.nb_classes + 1))
				weights += tmp
			weights = weights.astype(np.float32)
			weights = weights / np.sum(weights)
			self.weights = weights ** -0.5

		elif self.mode == 'val':
			self.weights = np.ones(self.nb_classes)

	@staticmethod
	def graipher(pts, n):
		'based on Grapher https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python'
		farthest_pts = np.zeros((n, pts.shape[1]))
		farthest_pts[0] = pts[np.random.randint(len(pts))]
		distances = ((farthest_pts[0,0:3] - pts[:,0:3]) ** 2).sum(axis=1)
		for i in range(1, n):
			farthest_pts[i] = pts[np.argmax(distances)]
			distances = np.minimum(distances, ((farthest_pts[i,0:3] - pts[:,0:3]) ** 2).sum(axis=1))
		return farthest_pts

	def __getitem__(self, index):

		if self.scale:
			data = np.copy(self.data_all[index])
			scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
			data[:, 0:3] = scaler.fit_transform(data[:, 0:3])
		else:
			data = self.data_all[index]

		label = self.label_all[index].astype(np.int32)
		weight = self.weights[label]

		return data, label, weight

	def __len__(self):
		return len(self.data_all)

if __name__=='__main__':

	root = '/'
	data = HessigheimDataset(root, scale=False, num_faces=15000, nb_classes=8, sampling='knn', mode='val')

	print('Number of batches: ', len(data))

	is_plot = True

	if is_plot:

		rgb_color = [[255, 255, 0], [128, 0, 0], [255, 0, 255], [0, 255, 0], [0, 128, 0], [0, 255, 255],
					 [255, 128, 0], [128, 128, 128]]

		for i in range(len(data)):
			faces, labels, weights = data[i]
			colors = np.zeros((faces.shape[0],3))

			for i in range(faces.shape[0]):
				ind = labels[i]
				colors[i, 0:3] = rgb_color[int(ind)]

			pcd_1 = open3d.PointCloud()
			pcd_1.points = open3d.Vector3dVector(faces[:, :3])
			pcd_1.colors = open3d.Vector3dVector(colors/255.)
			open3d.draw_geometries([pcd_1])

		#	fig = plt.figure()
		#	ax = fig.add_subplot(111, projection='3d')
		#	xs = faces[:, 0]
		#	ys = faces[:, 1]
		#	zs = faces[:, 2]

		#	ax.scatter(xs,ys,zs, c=colors/255.,  s=0.2)
		#	ax.set_xlim(-1,1)
		#	ax.set_ylim(-1,1)
		#	ax.set_zlim(-1,1)
		#	ax.set_xticks([-1,1])
		#	ax.set_yticks([-1,1])
		#	ax.set_zticks([-1,1])
		#	ax.w_xaxis.set_pane_color((0, 0, 0, 1))
		#	ax.w_yaxis.set_pane_color((0, 0, 0, 1))
		#	ax.w_zaxis.set_pane_color((0, 0, 0, 1))
		#	plt.show()