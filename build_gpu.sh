module load gcc/5.4.0
module load intel/2018.4
#module load openmpi/3.1.4
module load cuda/10.0
module load cuDNN/v7.6forcuda10.0
module load cmake

source ~/env/plumed261_rid.env
export CMAKE_PREFIX_PATH="/data2/publicsoft/fftw/3.3.8-f"
export CC=/data2/publicsoft/gcc/5.4.0/bin/gcc
export CXX=/data2/publicsoft/gcc/5.4.0/bin/g++


rm -rf build;
mkdir build;
cd build; 
cmake .. -DGMX_GPU=ON -DGMX_THREAD_MPI=ON -DCUDA_TOOLKIT_ROOT_DIR=/data2/publicsoft/cuda10.0 -DCMAKE_INSTALL_PREFIX=/data1/ddwang/software/gromacs-2020.2-sitsbias-gpu -DGMX_PREFER_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF -DGMX_EXTERNAL_BLAS=off && make -j install
