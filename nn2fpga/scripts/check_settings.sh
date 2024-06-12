#! /bin/bash 
if [ ! -d $PRJ_ROOT ]; then
    mkdir -p $PRJ_ROOT 
fi

if [ ! -d $PRJ_ROOT/cc/src ]; then 
  mkdir -p $PRJ_ROOT/cc/src
fi

if [ ! -d $PRJ_ROOT/cc/include ]; then 
  mkdir -p $PRJ_ROOT/cc/include
fi
