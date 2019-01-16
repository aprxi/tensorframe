#!/bin/sh

_err(){
  echo "######## BUILD FAILED ########"
  echo "$1"
  exit 1
}

_exec(){
  echo ${1};${1};r="$?";
  [ ! "$r" -eq 0 ] && _err "ERROR:$1"
}

build_cpplibs(){
#  _exec "make -C ./cpp libtensor"
#  _exec "make -C ./cpp libparser"
  git submodule update --recursive
  _exec "make -C ./cpp libio"
  _exec "make -C ./cpp libbinops"
  _exec "cp cpp/build/*.so tensorframe/core/libs/"
}

build_tensorframe(){
  _exec "python3 setup.py sdist bdist_wheel"
  _exec "sudo python3 setup.py install"
}

if [ -z "$1" ];then
  build_cpplibs
  build_tensorframe
elif [ "$1" = "cpplibs" ];then
  build_cpplibs
elif [ "$1" = "tensorframe" ];then
  build_tensorframe
else
  _err "Unknown build: $1"
fi

echo "######## BUILD SUCCESS ########"
exit 0
