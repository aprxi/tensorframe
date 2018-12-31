#!/bin/sh

if [ ! -z "$1" ];then
    python3 -m pylint test/test_${1}\.py
else
    python3 -m pylint test/test_*\.py
fi
