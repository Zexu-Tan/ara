#!/usr/bin/env bash

ROOT_DIR="$(pwd)"

if [ -z "$1" ]; then
  echo "Usage: $0 func"
  exit 1
fi

# Generate .dump file
cd $ROOT_DIR/apps 
make "$1" 
cd -

cd $ROOT_DIR/hardware
app=$1 make simc
cd -
