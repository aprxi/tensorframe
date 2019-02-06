#!/bin/sh

if [ ! -z "$1" ];then
    nosetests3 --config nose.cfg --nocapture test/test_${1}\.py
else
    nosetests3 --config nose.cfg test/test_*\.py
fi
