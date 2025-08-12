#!/usr/bin/env bash

# Change and display working directory
echo "${PWD}"
cd /scratch/saguilar/SAGE/
echo "${PWD}"
echo

python -V
echo

# Create virtual environment if it doesn't exist
if [ ! -d ./env/ ]; then
	python -m venv env
	source ./env/bin/activate
	pip install --no-cache-dir -U pip
	pip install --no-cache-dir -r requirements.txt
    python -m tmtoolkit setup en
	echo
else
	source ./env/bin/activate
fi

# Run access point with argument
echo "Starting script ..."
python main.py classification
echo "DONE!"