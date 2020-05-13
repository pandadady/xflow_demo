if [ -z $1 ]
then
    echo " please input command "
fi
if [ $1 == 'deploy' ]
then
    mkdir -p ../run/data
    mkdir -p ../run/model
    cp -r script ../run
    cp mapfeat.py ../run/mapfeat.py
    cp run* ../run
fi

if [ $1 == 'make' ]
then
    rm -rf build
    mkdir build
    cd build
    cmake ..
    make -j
    cd ..
    rm -rf ../run/build
    cp -r build  ../run
fi

