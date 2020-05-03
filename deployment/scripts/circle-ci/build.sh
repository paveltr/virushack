#!/usr/bin/env bash

set -o allexport
source ~/.bashrc
set +o allexport

set -e

# Import env vars
if [[ "$CIRCLE_BRANCH" == "develop" ]]; then
  APP_ENV=staging
elif [[ "$CIRCLE_BRANCH" == "master" ]]; then
  APP_ENV=production
else
  APP_ENV=issue
fi

# Import env vars
set -o allexport
source deployment/.env.${APP_ENV}
set +o allexport

PATH=$PATH:/home/circleci/google-cloud-sdk/bin

gcloud auth activate-service-account ${GCLOUD_EMAIL} --key-file ${GCLOUD_KEY}

gcloud docker --authorize-only

# Build flow
docker build -t ${CONTAINER_IMAGE_NAME}:${APP_ENV} -f ${DOCKER_FILE} --no-cache --pull .

echo "Build finished"
