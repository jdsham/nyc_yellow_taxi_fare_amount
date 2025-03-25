#!/bin/bash

cd "$(dirname "$0")"
while IFS= read -r script_path; do
    echo "Executing script: $script_path"
    python $script_path
done < "experiments.txt"