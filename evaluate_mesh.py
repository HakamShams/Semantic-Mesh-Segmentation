'''

Utility mesh function for mesh visualization

Author: Hakam Shams
Date: Novemebr 2019

Input:  test_file      : h5 file includes data and prediction of the tile, e.g. from predict_knn.py
        obj_file       : obj file
        class_names    : names of classes
        rgb_color      : color map for classes
        rgb_color_diff : color map for difference between predicted and GT labels

        is_plot        : plot confusion matrix
        is_visualize   : visualize the results

Dependencies: numpy - os - h5py - vtkplotter - pywavefront - matplotlib - scipy - sklearn

'''

import numpy as np
import os
import h5py
from vtkplotter import *
import vtkplotter as vtk
import pywavefront
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix

np.set_printoptions(suppress=True, precision=2)
settings.screeshotScale = 4

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Input:

test_file = '/'
obj_file = ''

is_plot = True
is_visualize = True

class_names = ['building', 'roof', 'impervious surface', 'green space', 'mid and high vegetation', 'vehicle',
               'chimney/antenna', 'clutter']

nb_classes = len(class_names)

rgb_color = [[255, 255, 255], [128, 0, 0], [128, 0, 128], [0, 255, 0],
            [0, 128, 0], [0, 255, 255],[255, 128, 0], [128, 128, 128]]

rgb_color_diff = [[255, 0, 0], [0, 255, 0]]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def main():

    file = h5py.File(test_file, mode='r')
    data = np.array(file['data'])
    Y_test = np.array(file['gt'], dtype=int)
    Y_pred = np.array(file['label_probability'], dtype=int)
    Y_cat = np.array(file['label_cat'])
    #Y_prob = np.array(file['label_prob'])

    file_name = os.path.splitext(test_file)[0]
    print('file name:', file_name)

    print('number of faces: ', data.shape[0])

    Y = np.array(Y_pred) - np.array(Y_test)

    for k in range(len(Y)):
        if Y[k] == 0:
            Y[k] = 1
        else:
            Y[k] = 0

    Y_pred = Y_pred.flatten().tolist()
    Y_test = Y_test.flatten().tolist()

    # compute mean IoU

    cm = confusion_matrix(Y_test, Y_pred)

    # compute mean iou
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    print('\nIoU: ', IoU)

    recall = np.diag(cm) / np.sum(cm, axis=1)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    print('\nrecall: ', recall)
    print('precision: ', precision)

    Y_pred = [str(i) for i in Y_pred]
    Y_test = [str(i) for i in Y_test]

    print('\nmean mIoU: ', np.mean(IoU))
    print('overall acc: ', np.diagonal(cm).sum()/np.matrix(cm).sum())
    #print(classification_report(Y_test, Y_pred, target_names=class_names))

    scene = pywavefront.Wavefront(
        obj_file,
        create_materials=True,
        collect_faces=True,
    )
    f = np.array(scene.mesh_list[0].faces)
    v = np.array(scene.vertices)

    def calc_distances(p0, p1):
        return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2)

    A = np.zeros(f.shape[0])
    C = np.zeros((f.shape[0], 3))

    for i in range(f.shape[0]):
        p1 = np.array([v[f[i, 0], 0], v[f[i, 0], 1], v[f[i, 0], 2]])
        p2 = np.array([v[f[i, 1], 0], v[f[i, 1], 1], v[f[i, 1], 2]])
        p3 = np.array([v[f[i, 2], 0], v[f[i, 2], 1], v[f[i, 2], 2]])

        C[i, :] = [(p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3, (p1[2] + p2[2] + p3[2]) / 3]

        # calculate the semi-perimeter
        l1 = calc_distances(p1, p2)
        l2 = calc_distances(p2, p3)
        l3 = calc_distances(p3, p1)

        s = (l1 + l2 + l3) / 2

        # calculate the area
        A[i] = np.sqrt(s * (s - l1) * (s - l2) * (s - l3))

    # consider just labeled faces

    tree_a = cKDTree(C[:, 0:3])

    ind_a = []

    for i in range(data.shape[0]):
        s, ind = tree_a.query(data[i, 0:3], k=1)
        ind_a.append(ind)

    A = np.take(A, ind_a, axis=0)

    A_correct = np.squeeze(np.argwhere(Y == 1), axis=-1)
    #A_not_correct = np.squeeze(np.argwhere(Y == 0), axis=-1)

    A_correct = A[A_correct.astype(np.int)].sum()
    #A_not_correct = A[A_not_correct.astype(np.int)].sum()

    print('\ncorrectly predicted surface area: ', A_correct / A.sum())
    #print(A_not_correct * 100 / A.sum())

    # averaged maximum label probability per face
    Y_p = np.zeros(len(Y_test))
    for i in range(Y_cat.shape[0]):
        Y_p[i] = np.max(Y_cat[i, :]) * 100/ Y_cat[i, :].sum()
    # number of predictions per face
    Y_n = np.zeros(len(Y_test))
    for i in range(Y_cat.shape[0]):
        Y_n[i] = Y_cat[i, :].sum()
    # number of classes per face
    Y_nb_c = np.zeros(len(Y_test))
    for i in range(Y_cat.shape[0]):
        Y_nb_c[i] = np.count_nonzero(Y_cat[i, :])

    if is_plot:

        def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):

            if not title:
                if normalize:
                    title = 'Normalized confusion matrix'
                else:
                    title = 'Confusion matrix, without normalization'

            cm = confusion_matrix(y_true, y_pred)

            s = np.sum(cm, axis=1)
            classes_y = []
            for i in range(len(classes)):
                classes_y.append(classes[i] + '\n' + str(s[i]) + ' = ' + str(np.round(100 * s[i]/np.matrix(cm).sum(),3)) + '%')

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("\nNormalized confusion matrix")
            else:
                print('\nConfusion matrix, without normalization')

            print(cm)

            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=classes, yticklabels=classes_y,
                   title=title,
                   ylabel='Ground Truth',
                   xlabel='Prediction')

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()

            return ax

        plot_confusion_matrix(Y_test, Y_pred, classes=class_names, normalize=True,
                              title='Normalized confusion matrix', cmap=plt.cm.Greens)
        plt.show()

    if is_visualize:

        tree = cKDTree(data[:, 0:3])

        color_test = np.array(rgb_color)[np.array(Y_test).astype(np.int)]
        color_pred = np.array(rgb_color)[np.array(Y_pred).astype(np.int)]
        color_diff = np.array(rgb_color_diff)[Y.astype(np.int)]

        cols_test, cols_pred, cols_diff, cols_y_n, cols_y_p, cols_y_nb_c = [], [], [], [], [], []

        for i in range(C.shape[0]):
            _, ind = tree.query(C[i, :], k=1)
            j_test = color_test[ind]
            cols_test.append(j_test)
            j_pred = color_pred[ind]
            cols_pred.append(j_pred)
            j_diff = color_diff[ind]
            cols_diff.append(j_diff)
            j_y_n = Y_n[ind]
            cols_y_n.append(j_y_n)
            j_y_p = Y_p[ind]
            cols_y_p.append(j_y_p)
            j_y_nb_c = Y_nb_c[ind]
            cols_y_nb_c.append(j_y_nb_c)

        mesh_test = vtk.load(obj_file)
        mesh_pred = vtk.load(obj_file)
        mesh_diff = vtk.load(obj_file)

        mesh_test.cellColors(cols_test, mode='colors')
        mesh_pred.cellColors(cols_pred, mode='colors')
        mesh_diff.cellColors(cols_diff, mode='colors')


        vtk.show([(mesh_test, Text('Ground Truth', s=1, pos='top-middle')),
                  (mesh_pred, Text('Prediction', s=1, pos='top-middle')),
                  (mesh_diff, Text('Difference', s=1, pos='top-middle'))], N=3, newPlotter=True, bg='beige', axes=0)

        #tk.show(mesh_test, Text('Ground Truth', s=1, pos='top-middle'), newPlotter=True, bg='beige', axes=0)
        #vtk.show(mesh_pred, Text('Prediction', s=1, pos='top-middle'), newPlotter=True, bg='beige', axes=0)
        #vtk.show(mesh_diff, Text('Difference', s=1, pos='top-middle'), newPlotter=True, bg='beige', axes=0)

        mesh_y_n = vtk.load(obj_file)
        mesh_y_nb_c = vtk.load(obj_file)
        mesh_y_p = vtk.load(obj_file)

        mesh_y_n.cellColors(cols_y_n, cmap="hsv")
        mesh_y_n.addScalarBar(c='white', nlabels=7, title="Predictions per face")
        mesh_y_nb_c.cellColors(cols_y_nb_c, cmap="jet")
        mesh_y_nb_c.addScalarBar(c='w', vmin=1, nlabels=int(np.max(cols_y_nb_c)), title="Classes per face")
        mesh_y_p.cellColors(cols_y_p, cmap="viridis")
        mesh_y_p.addScalarBar(c='w', title="Class Probability %")

        vtk.show(mesh_y_n, Text('Predictions per face', s=1, pos='top-middle'), newPlotter=True,
                 axes=0, bg='beige')

        vtk.show([(mesh_y_nb_c, Text('Classes per face', s=1, pos='top-middle')),
                  (mesh_y_p, Text('Class Probability %', s=1, pos='top-middle'))],
                 N=2, newPlotter=True, bg='beige')


if __name__ == '__main__':
    main()
