#!/bin/bash

APPDIR=/var/www/src
UWSGI_CONFIG_DIR=/var/www/bin
SOCKFILE=/var/run/sle/sle.sock
NGINX_PID=/var/run/sle/nginx.pid

export PYTHONPATH=$APPDIR:$PYTHONPATH
mkdir -p $(dirname $SOCKFILE)

$CONDA_ACTIVATE

if [ "$ENVIRONMENT" == "development" ]; then
    uwsgi --ini $UWSGI_CONFIG_DIR/local_sle_uwsgi.ini
else
    uwsgi --ini $UWSGI_CONFIG_DIR/sle_uwsgi.ini
fi
