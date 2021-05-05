# Changes:
# 1. GMX_GPU set from ON to CUDA
# 2. gcc version from 5.4.0 to 7.4.0 (later than 7 is required)

GCC_VERSION='7.4.0'
CUDA_VERSION='10.0'

module load gcc/$GCC_VERSION
module load intel/2018.4
#module load openmpi/3.1.4
module load cuda/$CUDA_VERSION
module load cuDNN/v7.6forcuda$CUDA_VERSION
module load cmake/3.20.0

export LD_LIBRARY_PATH=/data1/anguse/zijian/plumed2/lib:$LD_LIBRARY_PATH
export PATH=/data1/anguse/zijian/plumed2/bin:$PATH
#export LD_LIBRARY_PATH=/data1/anguse/local/plumed-gmx2020/lib:$LD_LIBRARY_PATH
#export PATH=/data1/anguse/local/plumed-gmx2020/bin:$PATH
export CMAKE_PREFIX_PATH="/data2/publicsoft/fftw/3.3.8-f"
export CC=/data2/publicsoft/gcc/$GCC_VERSION/bin/gcc
export CXX=/data2/publicsoft/gcc/$GCC_VERSION/bin/g++

rm -rf build;
mkdir build;
cd build; 
cmake .. -DGMX_MPI=ON -DGMX_GPU=CUDA -DCUDA_TOOLKIT_ROOT_DIR=/data2/publicsoft/cuda10.0 -DCMAKE_INSTALL_PREFIX=/data1/anguse/yuxuan/bin/gromacs2021-fep -DREGRESSIONTESTS_DOWNLOAD=OFF -DREGRESSIONTEST_PATH=/data1/anguse/yuxuan/regressiontests-2021.1 && make -j 8 &&  make check && make install
