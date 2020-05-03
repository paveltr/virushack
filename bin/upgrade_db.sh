#!/bin/bash

$CONDA_ACTIVATE

cd "$MAIN_PATH/src"
export FLASK_APP="sle"
flask db upgrade