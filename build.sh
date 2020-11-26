module load gcc/5.4.0
module load intel/2018.4
#module load openmpi/3.1.4
module load cuda/10.0
module load cuDNN/v7.6forcuda10.0
module load cmake

INSTALL_DIR=/data1/anguse/zijian/gromacs-2020.2-gpuall-softbond

export LD_LIBRARY_PATH=/data1/anguse/zijian/plumed2/lib:$LD_LIBRARY_PATH
export PATH=/data1/anguse/zijian/plumed2/bin:$PATH
#export LD_LIBRARY_PATH=/data1/anguse/local/plumed-gmx2020/lib:$LD_LIBRARY_PATH
#export PATH=/data1/anguse/local/plumed-gmx2020/bin:$PATH
export CMAKE_PREFIX_PATH="/data2/publicsoft/fftw/3.3.8-f"
export CC=/data2/publicsoft/gcc/5.4.0/bin/gcc
export CXX=/data2/publicsoft/gcc/5.4.0/bin/g++

rm -rf build license.c license*.so;
mkdir build;
cd build; 
cmake .. -DGMX_MPI=ON -DGMX_GPU=ON -DCUDA_TOOLKIT_ROOT_DIR=/data2/publicsoft/cuda10.0 -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DPYTHON_HOME=/data2/publicsoft/anaconda3 -DPYTHON_LIBRARIES=/data2/publicsoft/anaconda3/lib/libpython3.7m.so && make -j16 && make install
cd ..
python cythonize.py build_ext --inplace
cp license*.so $INSTALL_DIR/lib64
