rm -rf build
mkdir build
cd build
cmake ..
make -j
cd ..
rm -rf ../run/build
cp -r build  ../run
