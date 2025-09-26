#!/usr/bin/env bash

ROOT_DIR="$(pwd)"

export kernel_undertest="$1"

if [ -z "$1" ]; then
  echo "Usage: $0 func"
  exit 1
fi


cd "$ROOT_DIR/apps/bin"
rm -f "$1"
rm -f "$1.dump"


cd "$ROOT_DIR/apps/$1"
find . -name "*.o" -type f -delete
find . -name "*.o.spike" -type f -delete

cd "$ROOT_DIR/apps/$1/kernel"
find . -name "*.o" -type f -delete
find . -name "*.o.spike" -type f -delete



# Generate .dump file
cd "$ROOT_DIR/apps"
make "bin/$1"

cd "$ROOT_DIR/apps/bin/"
cp "$1.dump" "$ROOT_DIR/gen_replaced_trace/$1.dump"

cd "$ROOT_DIR/hardware"
app=$1 make sim


cd "$ROOT_DIR/hardware/build"

cp "trace_hart_0.log" "$ROOT_DIR/hardware/vector_traces/"


 
cd "$ROOT_DIR/apps/bin"

cp "$1.dump" "$ROOT_DIR/hardware/vector_traces/"



cd "$ROOT_DIR"

mkdir -p "gen_replaced_trace"


cd "$ROOT_DIR/hardware/"

python3 "validate_vector.py"

cp "trace_hart_0_plus_vinstr.txt" "$ROOT_DIR/gen_replaced_trace/$1.txt"
