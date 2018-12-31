#!/bin/sh

#!/bin/sh

_err(){
  echo "######## CLEAN FAILED ########"
  echo "$1"
  exit 1
}

_exec(){
  echo ${1};${1};r="$?";
  [ ! "$r" -eq 0 ] && _err "ERROR:$1"
}

clean_cpplibs(){
	_exec "make -C ./cpp clean"
}

clean_tensorframe(){
    _exec "rm -rf ./test/__pycache__ ./tensorframe/__pycache__ ./tensorframe/core/__pycache__ ./tensorframe/core/libs/__pycache__"
	_exec "rm -rf ./build ./dist ./tensorframe.egg-info"
	_exec "rm -rf ./tensorframe/core/libs/*pyc ./tensorframe/core/libs/*so"
}

if [ -z "$1" ];then
  clean_tensorframe
  clean_cpplibs
elif [ "$1" = "cpplibs" ];then
  clean_cpplibs
elif [ "$1" = "tensorframe" ];then
  clean_tensorframe
else
  _err "Unknown clean: $1"
fi

echo "######## CLEAN SUCCESS ########"
exit 0
