'''

Utility mesh function for LiDAR statistics

Author: Hakam Shams
Date: Novemebr 2019

Input:  label_file     : LiDAR Points with its corresponding Face Id from mesh e.g. from LiDAR.py
        threshold_file : Faces of mesh with its corresponding used threshold e.g. from LiDAR.py
        test_file      : h5 file includes data and prediction of the tile, e.g. from evaluation_knn.py
        obj_file       : obj file

        k_voting       : k nearest neighborhood to label the entire point cloud
        class_names    : names of classes
        area_split     : splits of area for histogram over the surface area per class
        rgb_color      : color map for classes
        rgb_color_diff : color map for the difference between predicted/mapped and GT
        rgb_color_t    : color map for thresholds

        is_Histogram       : plot histograms
        is_Mesh            : visualise the mesh
        is_LiDAR_points    : visualise LiDAR points
        is_LiDAR_confusion : plot LiDAR confusion matrix

Code lines of 'Statistic over the used thresholds per face for the association of LiDAR point cloud and mesh'
is written w.r.t three thresholds.

Dependencies: numpy - h5py - scipy - sklearn - vtkplotter - pywavefront

'''

import numpy as np
import h5py
import vtkplotter as vtk
from vtkplotter import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pywavefront

settings.screeshotScale = 8
np.set_printoptions(precision=2)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Input:

is_Histogram = True
is_LiDAR_points = False
is_LiDAR_confusion = False
is_Mesh = True

k_voting = 50

label_file = '/'
threshold_file = '/'
test_file = '/'
obj_file = '/'

class_names = ['building','roof','impervious surface','green space','mid and high vegetation','vehicle','chimney/antenna','clutter']
nb_classes = len(class_names)

rgb_color = [[245, 245, 245], [128, 0, 0], [128, 0, 128], [0, 255, 0], [0, 128, 0], [0, 255, 255], [255, 128, 0],
             [128, 128, 128]]

rgb_color_diff = [[0, 255, 0], [255, 0, 0]]

rgb_color_t = [[224, 255, 255], [64, 224, 208], [0, 139, 139], [47, 79, 79]]

area_split = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 12.0]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Algorithm:

if label_file.endswith('h5'):
    hdf = h5py.File(label_file, mode='r')
    data = np.array(hdf['data'])
elif label_file.endswith('txt'):
    data = np.loadtxt(label_file)

if threshold_file.endswith('h5'):
    hdf = h5py.File(threshold_file, mode='r')
    threshold = np.array(hdf['data'])
elif threshold_file.endswith('txt'):
    threshold = np.loadtxt(threshold_file)

print('Total number of LiDAR points: ', data.shape[0])

index = np.squeeze(np.argwhere(data[:,-1] > -1),axis=-1)

hdf = h5py.File(test_file, mode='r')
Y_test_face = np.array(hdf['gt']).astype(np.int32)
Y_pred_face = np.array(hdf['label_probability']).astype(np.int32)
data_face = np.array(hdf['data'])

print('\nTotal number of labeled mesh faces: ', Y_test_face.shape[0])

y_all = []

for i in range(8):

    y_all.append(np.argwhere(Y_test_face == i).shape[0])

scene = pywavefront.Wavefront(
        obj_file,
        create_materials=True,
        collect_faces=True)

F = np.array(scene.mesh_list[0].faces)
V = np.array(scene.vertices, dtype='float64')
print("Number of Faces: ", np.array(scene.mesh_list[0].faces).shape)
print("Number of Vertices: ", np.array(scene.vertices).shape)

C = np.zeros((F.shape[0],3))
A = np.zeros(F.shape[0])

def calc_distances(p0, p1):
    return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2)

for i in range(F.shape[0]):
    p1 = np.array([V[F[i, 0], 0], V[F[i, 0], 1], V[F[i, 0], 2]])
    p2 = np.array([V[F[i, 1], 0], V[F[i, 1], 1], V[F[i, 1], 2]])
    p3 = np.array([V[F[i, 2], 0], V[F[i, 2], 1], V[F[i, 2], 2]])

    v1 = p3 - p1
    v2 = p2 - p1

    C[i, :] = [(p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3, (p1[2] + p2[2] + p3[2]) / 3]

    # calculate the semi-perimeter
    l1 = calc_distances(p1, p2)
    l2 = calc_distances(p2, p3)
    l3 = calc_distances(p3, p1)
    s = (l1 + l2 + l3) / 2

    # calculate the area
    A[i] = np.sqrt(s * (s - l1) * (s - l2) * (s - l3))

tree = cKDTree(C[:,0:3])

index_labeled = []

for i in range(data_face.shape[0]):
    _, ind = tree.query(data_face[i, 0:3], k=1)
    index_labeled.append(ind)

A = np.take(A,index_labeled,axis=0)

Y_test = np.copy(data[:,-1])
indices = np.array(threshold[:,-1])
indices = np.squeeze(np.argwhere(indices!=-1),axis=-1)

indices_not = np.isin(np.arange(Y_test_face.shape[0]),indices)
indices_not = np.squeeze(np.where(indices_not == False),axis=0)

print('\nNumber of detected faces: ', np.array(indices).shape[0])
print('Ratio of detected faces: ' + str(np.array(indices).shape[0] * 100/Y_test_face.shape[0]) + '%')

print('\nNumber of not detected faces: ', np.array(indices_not).shape[0])
print('Ratio of not detected faces: ' + str(np.array(indices_not).shape[0] * 100/Y_test_face.shape[0]) + '%')

Y_detected = Y_test_face[indices.astype(np.int)]

Y_not_detected = Y_test_face[indices_not.astype(np.int)]

y_all_detected = []

for i in range(nb_classes):
    y_all_detected.append(np.argwhere(Y_detected == i).shape[0])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

if is_Histogram:

    # Per-class histogram of faces with/without associated LiDAR features
    y = np.array((y_all) - np.array(y_all_detected)) * 100/ np.array(y_all)
    p1 = plt.bar(range(nb_classes), y, 0.4, color='powderblue')
    p2 = plt.bar(range(nb_classes), np.array(y_all_detected) * 100/ np.array(y_all), 0.4, bottom=y, color='cadetblue')
    plt.ylabel('% Face')
    plt.title('LiDAR features detection per class')

    class_names_ratio = []
    for i, v in enumerate(class_names):
        class_names_ratio.append(v + '\n(' + str(y_all[i]) + ' faces)')

    plt.xticks(range(nb_classes), class_names_ratio, rotation=20, ha="right", rotation_mode="anchor")

    plt.yticks(np.arange(0, 110, 10))
    plt.legend((p2[0], p1[0]), ('with LiDAR features (' + str(np.round(np.array(indices).shape[0] * 100/Y_test_face.shape[0],2).astype(np.float)) + '%' + ')',
                                'without LiDAR features (' + str(np.round(np.array(indices_not).shape[0] * 100/Y_test_face.shape[0], 2).astype(np.float)) + '%' + ')'),loc='upper right')

    def autolabel(rects_1, rects_2):
        for rect, rect_2 in zip(rects_1, rects_2):
            height = np.round(rect.get_height(),2)
            plt.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', color='darkred')

            height_2 = np.round(rect_2.get_height(), 2)
            plt.annotate('{}'.format(height_2),
                         xy=(rect_2.get_x() + rect_2.get_width() / 2, (height_2 / 2) + height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', color='darkgreen')

    autolabel(p1, p2)

    plt.show()

    # Per-class histogram of surface area with/without associated LiDAR features
    A_detected = A[indices.astype(np.int)]
    A_not_detected = A[indices_not.astype(np.int)]

    a_all, a_all_detected, a_all_not = [], [], []

    for i in range(nb_classes):

        a_all.append(A[np.argwhere(Y_test_face == i)].sum())
        a_all_detected.append(A_detected[np.argwhere(Y_detected == i)].sum())
        a_all_not.append(A_not_detected[np.argwhere(Y_not_detected == i)].sum())

    print('Ratio of detected Area: ' + str(np.array(a_all_detected).sum() * 100/np.array(a_all).sum()) + '%')
    print('Ratio of not detected Area: ' + str(np.array(a_all_not).sum() * 100/np.array(a_all).sum()) + '%')

    a_all = np.round(a_all).astype(np.int)
    a_all_detected = np.round(a_all_detected).astype(np.int)

    a = np.array((a_all) - np.array(a_all_detected)) * 100/ np.array(a_all)
    p3 = plt.bar(range(nb_classes), a, 0.4, color='lightsteelblue')
    p4 = plt.bar(range(nb_classes), np.array(a_all_detected) * 100/ np.array(a_all), 0.4, bottom=a, color='lightslategray')
    plt.ylabel('% Surface area')
    plt.title('LiDAR features detection per class')

    class_names_ratio_a = []
    for i, v in enumerate(class_names):
        class_names_ratio_a.append(v + '\n(' + str(a_all[i]) + ' m$^2$)')

    plt.xticks(range(nb_classes), class_names_ratio_a, rotation=20, ha="right", rotation_mode="anchor")

    plt.yticks(np.arange(0, 110, 10))
    plt.legend((p4[0], p3[0]), ('with LiDAR features (' + str(np.round(np.array(a_all_detected).sum() * 100/np.array(a_all).sum(),2).astype(np.float)) + ' %' + ')',
                                'without LiDAR features (' + str(np.round(np.array(a_all_not).sum() * 100/np.array(a_all).sum(),2).astype(np.float)) + ' %' + ')'),loc='upper right')

    autolabel(p3, p4)

    plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    # Histogram over the surface area per class
    A_detected = A[indices.astype(np.int)]
    A_not_detected = A[indices_not.astype(np.int)]

    for l in range(nb_classes):
        b_all, b_all_detected, b_all_not = [], [], []

        for i in range(len(area_split)-1):

            b_1 = A[np.argwhere(Y_test_face == l)]
            b_2 = A_detected[np.argwhere(Y_detected == l)]
            b_3 = A_not_detected[np.argwhere(Y_not_detected == l)]

            b_all.append(b_1[np.logical_and((area_split[i] <= b_1),(b_1 < area_split[i+1]))].sum())
            b_all_detected.append(b_2[np.logical_and((area_split[i] <= b_2),(b_2 < area_split[i+1]))].sum())
            b_all_not.append(b_3[np.logical_and((area_split[i] <= b_3),(b_3 < area_split[i+1]))].sum())

        b = np.array((b_all) - np.array(b_all_detected)) * 100/ np.array(b_all).astype(np.float)
        p5 = plt.bar(range(len(area_split)-1), b, 0.4, color='lightsteelblue')
        p6 = plt.bar(range(len(area_split)-1), np.array(b_all_detected) * 100/ np.array(b_all).astype(np.float), 0.4, bottom=b, color='lightslategray')
        plt.ylabel('% Surface area')
        plt.title('LiDAR features association (' + str(class_names[l]) +')')

        area_ticks = []

        for i in range(len(area_split[:-1])):
            area_ticks.append('('+ str(area_split[i]) + '-' + str(area_split[i+1]) + ') m$^2$')

        plt.xticks(range(len(area_split)-1),area_ticks,rotation=20, ha="right", rotation_mode='anchor')

        plt.yticks(np.arange(0, 110, 10))

        plt.legend((p6[0], p5[0]), ('with LiDAR features','without LiDAR features'),loc='upper right')

        def autolabel(rects_1, rects_2):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect, rect_2 in zip(rects_1, rects_2):
                height = np.round(rect.get_height(), 2)
                plt.annotate('{}'.format(height),
                             xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom', color='darkred')

                height_2 = np.round(rect_2.get_height(), 2)
                plt.annotate('{}'.format(height_2),
                             xy=(rect_2.get_x() + rect_2.get_width() / 2, (height_2 / 2) + height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom', color='darkgreen')

        autolabel(p5, p6)

        plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    # Statistic over the used thresholds per face for the association of LiDAR point cloud and mesh
    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))

    t = np.array(threshold[:,-1])
    t_0 = np.squeeze(np.argwhere(t==-1),axis=-1)
    t_1 = np.squeeze(np.argwhere(t==0),axis=-1)
    t_2 = np.squeeze(np.argwhere(t==1),axis=-1)
    t_3 = np.squeeze(np.argwhere(t==2),axis=-1)

    t_name = ['without feature','first threshold','second threshold','third threshold']
    t_data = [len(t_0),len(t_1),len(t_2),len(t_3)]
    explode = (0.1, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'first threshold')

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.2f}%\n({:d} face)".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(t_data, autopct=lambda pct: func(pct, t_data),
                                      colors=['lightcyan','turquoise','darkcyan','darkslategray'],
                                      textprops=dict(color="black"),
                                      labels=t_name,
                                      explode=explode)

    t_name_cm = ['without feature','first threshold (+5 , -20 cm)','second threshold (+10 , -40 cm)','third threshold (+15 , -80 cm)']

    ax.legend(wedges, t_name_cm,
              loc="upper left",
              bbox_to_anchor=(1, 0, 0.5, 1),facecolor=[160/255,160/255,160/255])

    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("LiDAR features detection per threshold")

    plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

if is_LiDAR_points or is_LiDAR_confusion:

    # detected LiDAR points

    index = np.where(Y_test > -1)

    Y_pred = np.copy(data[:,-1])

    data_tree = np.copy(data[index])

    print('\nNumber of detected points: ', np.array(index).shape[1])
    print('Ratio of detected points: ' + str(np.array(index).shape[1] * 100/data.shape[0]) + ' %')

    Y_test[index] = Y_test_face[Y_test[index].astype(np.int32)]
    Y_pred[index] = Y_pred_face[Y_pred[index].astype(np.int32)]

    Y_test_all = np.copy(Y_test)
    Y_pred_all = np.copy(Y_pred)

    Y_test = Y_test[index]
    Y_pred = Y_pred[index]

    if is_LiDAR_points:

        colors_lidar_test = np.array(rgb_color)[np.array(Y_test).astype(np.int)]
        colors_lidar_pred = np.array(rgb_color)[np.array(Y_pred).astype(np.int)]

        lidar_p_test = Points(np.squeeze(data[index,0:3],axis=0), r=4, c='red')
        lidar_p_pred = Points(np.squeeze(data[index,0:3],axis=0), r=4, c='red')

        lidar_p_test.cellColors(colors_lidar_test, mode='colors')
        lidar_p_pred.cellColors(colors_lidar_pred, mode='colors')

        vtk.show(lidar_p_test, Text('GT of Detected LiDAR Points', s=1, pos='top-middle'), newPlotter=True, axes=0, bg='beige')
        vtk.show(lidar_p_pred, Text('Predection of Detected LiDAR Points', s=1, pos='top-middle'), newPlotter=True, axes=0, bg='beige')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # label missing LiDAR points

    tree = cKDTree(np.copy(data_tree[:,:-1]))
    indices_2 = np.argwhere(Y_test_all == -1)
    indices_2 = np.squeeze(indices_2, axis=-1)

    # adding indices of faces to missing points
    # nearest neighbor
    # Y_test_all[indices_2] = Y_test[ind_lidar].astype(np.int)
    # Y_pred_all[indices_2] = Y_pred[ind_lidar].astype(np.int)

    # majority voting
    for i in indices_2:

        _, ind_lidar = tree.query(data[i, 0:3], k=k_voting)

        Y_test_all[i] = np.argmax(np.bincount(Y_test[ind_lidar].astype(np.int)))
        Y_pred_all[i] = np.argmax(np.bincount(Y_pred[ind_lidar].astype(np.int)))

    if is_LiDAR_points:

        colors_lidar_test_all = np.array(rgb_color)[np.array(Y_test_all).astype(np.int)]
        colors_lidar_pred_all = np.array(rgb_color)[np.array(Y_pred_all).astype(np.int)]

        lidar_p_test_all = Points(data[:,0:3], r=4, c='red')
        lidar_p_pred_all = Points(data[:,0:3], r=4, c='red')

        lidar_p_test_all.cellColors(colors_lidar_test_all, mode='colors')
        lidar_p_pred_all.cellColors(colors_lidar_pred_all, mode='colors')

        vtk.show(lidar_p_test_all, Text('Ground Truth LiDAR Points', s=1, pos='top-middle'), newPlotter=True, axes=0, bg='beige')
        vtk.show(lidar_p_pred_all, Text('Prediction LiDAR Points', s=1, pos='top-middle'), newPlotter=True, axes=0, bg='beige')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # difference LiDAR points

    Y = np.array(Y_pred) - np.array(Y_test)
    Y_all = np.array(Y_pred_all) - np.array(Y_test_all)

    Y[np.argwhere(np.logical_or((Y>0),(Y<0)))] = 1
    Y_all[np.argwhere(np.logical_or((Y_all>0),(Y_all<0)))] = 1

    if is_LiDAR_points:

        color_diff = np.array(rgb_color_diff)[Y.astype(np.int32)]
        color_diff_all = np.array(rgb_color_diff)[Y_all.astype(np.int32)]

        lidar_p_diff = Points(np.squeeze(data[index,0:3],axis=0), r=4, c='red')
        lidar_p_diff_all = Points(data[:,0:3], r=4, c='red')

        colors_lidar_diff = np.array(rgb_color_diff)[np.array(Y).astype(np.int)]
        colors_lidar_diff_all = np.array(rgb_color_diff)[np.array(Y_all).astype(np.int)]

        lidar_p_diff.cellColors(colors_lidar_diff, mode='colors')
        lidar_p_diff_all.cellColors(colors_lidar_diff_all, mode='colors')

        vtk.show(lidar_p_diff, Text('Diff Ground Truth LiDAR Points', s=1, pos='top-middle'), newPlotter=True, axes=0, bg='beige')
        vtk.show(lidar_p_diff_all, Text('Diff Prediction LiDAR Points', s=1, pos='top-middle'), newPlotter=True, axes=0, bg='beige')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # confusion matrix LiDAR points

    Y_pred = Y_pred.flatten().tolist()
    Y_test = Y_test.flatten().tolist()

    Y_pred_all = Y_pred_all.flatten().tolist()
    Y_test_all = Y_test_all.flatten().tolist()

    cm = confusion_matrix(Y_test, Y_pred)
    cm_all = confusion_matrix(Y_test_all, Y_pred_all)

    recall = np.diag(cm) / np.sum(cm, axis=1)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    print('\noverall acc of detected LiDAR points: ', np.diagonal(cm).sum() / np.matrix(cm).sum())
    print('recall of detected LiDAR points: ', recall)
    print('precision detected: ', precision)

    recall_all = np.diag(cm_all) / np.sum(cm_all, axis=1)
    precision_all = np.diag(cm_all) / np.sum(cm_all, axis=0)
    print('\noverall acc all LiDAR points: ', np.diagonal(cm_all).sum() / np.matrix(cm_all).sum())
    print('recall all LiDAR points: ', recall_all)
    print('precision all LiDAR points: ', precision_all)

    # compute mean iou
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    print('\nIoU of detected LiDAR points: ', IoU)
    print('mIoU of detected LiDAR points: ', np.mean(IoU))

    intersection = np.diag(cm_all)
    ground_truth_set = cm_all.sum(axis=1)
    predicted_set = cm_all.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    print('\nIoU of all LiDAR points: ', IoU)
    print('mIoU of all LiDAR points: ', np.mean(IoU))

    Y_test = [str(i) for i in Y_test]
    Y_pred = [str(i) for i in Y_pred]

    Y_test_all = [str(i) for i in Y_test_all]
    Y_pred_all = [str(i) for i in Y_pred_all]

    if is_LiDAR_confusion:

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

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()

            return ax

        # Plot normalized confusion matrix
        plot_confusion_matrix(Y_test, Y_pred, classes=class_names, normalize=True,
                              title='Normalized confusion matrix\ndetected points', cmap=plt.cm.Greens)

        plot_confusion_matrix(Y_test_all, Y_pred_all, classes=class_names, normalize=True,
                              title='Normalized confusion matrix\nall points', cmap=plt.cm.Greens)

        plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if is_Mesh:

    # visualized detected faces

    mesh = vtk.load(obj_file)
    mesh.lc(lineColor='black')

    # threshold colors
    colors_t = np.ones((Y_test_face.shape[0],3))
    colors_t[:,:] = np.copy(np.array(rgb_color_t)[(threshold[:,-1]+1).astype(np.int)])

    cols_threshold = []

    tree = cKDTree(data_face[:,0:3])

    for i in range(C.shape[0]):
        _, ind = tree.query(C[i, 0:3], k=1)
        cols_threshold.append(colors_t[int(ind)])

    mesh.cellColors(cols_threshold, mode='colors')

    vtk.show(mesh, Text('Detected faces', s=1, pos='top-middle'), newPlotter=True, axes=0, bg='beige')