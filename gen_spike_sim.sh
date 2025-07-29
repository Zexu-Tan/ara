#!/usr/bin/env bash

ROOT_DIR="$(pwd)"

if [ -z "$1" ]; then
  echo "Usage: $0 func"
  exit 1
fi

cd $ROOT_DIR/apps 
make "bin/$1"
make "bin/$1.spike"
make "spike-run-$1"
cd -

