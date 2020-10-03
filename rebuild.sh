module load gcc/5.4.0
module load intel/2018.4
#module load openmpi/3.1.4
module load cuda/10.0
module load cuDNN/v7.6forcuda10.0
module load cmake

export LD_LIBRARY_PATH=/data1/anguse/zijian/plumed2/lib:$LD_LIBRARY_PATH
export PATH=/data1/anguse/zijian/plumed2/bin:$PATH
#export LD_LIBRARY_PATH=/data1/anguse/local/plumed-gmx2020/lib:$LD_LIBRARY_PATH
#export PATH=/data1/anguse/local/plumed-gmx2020/bin:$PATH
export CMAKE_PREFIX_PATH="/data2/publicsoft/fftw/3.3.8-f"
export CC=/data2/publicsoft/gcc/5.4.0/bin/gcc
export CXX=/data2/publicsoft/gcc/5.4.0/bin/g++

if [ -e Plumed.h ];then
plumed patch -r << EOF
2
EOF
fi;

plumed patch -p << EOF
2
EOF

cd build; 

make -j16 && make install
