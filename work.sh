if [ -z $1 ]
then
    echo " please input command "
    exit -1
fi
if [ $1 == 'deploy' ]
then
    mkdir -p ../run/data
    mkdir -p ../run/model
    cp -r scripts ../run
    cp mapfeat.py ../run/mapfeat.py
    cp run* ../run
    cp kill.sh ../run
    cp -r data ../run
else
  if [ $1 == 'make' ]
  then
      mkdir -p ../run/
      rm -rf build
      mkdir build
      cd build
      cmake ..
      make -j
      cd ..
      rm -rf ../run/build
      cp -r build  ../run
  else
      echo "input command error "
  fi
fi
