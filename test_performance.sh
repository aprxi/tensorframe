#!/bin/sh
# TODO: integrate output validation
if [ ! -z "$1" ];then
    nosetests3 --config nose.cfg --nocapture test/test_timeit_${1}\.py
fi
