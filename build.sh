
mkdir build
cd build
cmake ..
make -j
rm -rf ../run/build
cp -r build  ../run
