## Semantic Mesh Segmentation using PointNet/PointNet++

The aim of this project is to explore a semantic segmentation of meshes in an outdoor urban scenario and to make use of PointNet/PointNet++, which are point based deep neural networks applicable to meshes. The implemented pipeline segments meshes semantically including pre-processing, incorporation of LiDAR point cloud with mesh, training and evaluation. The code repository is the source code of the Master thesis (Semantic Mesh Segmentation using PointNet++) at the <a href="https://www.ifp.uni-stuttgart.de/lehre/masterarbeiten/602-Shams/">Institute of Photogrammetry (Ifp)</a>, University of Stuttgart, Germany. The used dataset is a 2.5 D textured mesh covering an urban area of village Hessigheim in Germany. The original implementation of PointNet++ can be found in <a href="https://github.com/charlesq34/pointnet2">GitHub</a> by Charles R Qi et al, from Stanford University.

![Example Prediction](Tile-C-PointNet++.png "Textured Mesh (top left), Ground Truth (top right), Prediction (bottom left), Differences (bottom right)")

![Example Mapping](Tile-A-Mapped-GT.jpg "Mapped ground truth from Mesh to LiDAR")

### Installation
The code is tested under TF1.9 GPU version and Python 3.7 on Ubuntu 16.04. Dependencies for Python libraries like `numpy`, `h5py`, `os`, `h5py`, `sys`, `vtkplotter`, `pywavefront`, `matplotlib`, `scipy`, `sklearn`, `open3d`, etc. PointNet++ requires that you have access to GPUs, while PointNet does not.

#### Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.2.0. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.

First, find Tensorflow include and library paths.

        TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
        TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
        
Then, add flags of `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands (see the source code <a href="https://github.com/charlesq34/pointnet2">GitHub</a>).

For example to compile customized TF operators in sampling folder under TF 1.9 and Cuda 9.0 using conda env:

1- open the file tf_sampling_compile.sh

2- open the folder tf_ops/sampling in the terminal

3- write the commands from file tf_sampling_compile.sh in the terminal:

        conda activate (your conda env)
        TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
        TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
        g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda-9.0/include -lcudart -L /usr/local/cuda-9.0/lib64/ -O2 -I $TF_INC/external/nsync/public -L $TF_LIB -ltensorflow_framework

4- test the compiling by running tf_sampling.py

### Usage

#### Preprocessing
First we split the training dataset into overlapped tiles:

       data_split.py
       
Then from these tiles we generate batches using different sampling methods. To check the batches generation:

       batch_prep.py

Additional preprocessing option to incorporate LiDAR features to mesh:

      LiDAR.py
      
#### Training
To train model based on PointNet/PointNet++:

      train.py
      
#### Evaluation
After training to generate predictions and evaluate the results e.g. based on kNN method:

      predict_knn.py
      evaluate_mesh.py

A visualization tool can be found in `evaluate_mesh.py` and `statistic_LiDAR.py`

Simple grid search for Random forest classifier can be found in `Random_Forest.py`

### License
The code is released under MIT License (see LICENSE file for details).

### Related Projects

* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation). Code and data released in <a href="https://github.com/charlesq34/pointnet">GitHub</a>.
* <a href="http://stanford.edu/~rqi/pointnet2/" target="_blank">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017) A hierarchical feature learning framework on point clouds. The PointNet++ architecture applies PointNet recursively on a nested partitioning of the input point set. It also proposes novel layers for point clouds with non-uniform densities.
* <a href="https://github.com/lwiniwar/alsNet" target="_blank">alsNet: Classification of 3D Point Clouds using Deep Neural Networks</a> by Lukas Winiwarter, TU Wien.

### Related Papers

* <a href="https://www.dgpf.de/src/tagung/jt2020/proceedings/proceedings/papers/27_DGPF2020_Laupheimer_et_al.pdf" target="_blank">The Importance of Radiometric Feature Quality for Semantic Mesh Segmentation</a> by D. Laupheimer, M. H. Shams Eddin, N. Haala. 40. Wissenschaftlich-Technische Jahrestagung der DGPF in Stuttgart – Publikationen der DGPF, Band 29, 2020, pp. 205-218.
* <a href="https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2020/509/2020/isprs-annals-V-2-2020-509-2020.pdf" target="_blank">On The Association Of Lidar Point Clouds And Textured Meshes For Multi-ModaL Semantic Segmentation</a> by D. Laupheimer, M. H. Shams Eddin, N. Haala. ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci., V-2-2020, 509–516, https://doi.org/10.5194/isprs-annals-V-2-2020-509-2020, 2020.
