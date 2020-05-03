#!/usr/bin/env bash

set -e

# Import env vars
if [[ "$CIRCLE_BRANCH" == "develop" ]]; then
  APP_ENV=staging
elif [[ "$CIRCLE_BRANCH" == "master" ]]; then
  APP_ENV=production
else
  APP_ENV=issue
  exit 0
fi

# Import env vars
set -o allexport
source deployment/.env.${APP_ENV}
set +o allexport

PATH=$PATH:/home/circleci/google-cloud-sdk/bin

gcloud auth activate-service-account ${GCLOUD_EMAIL} --key-file ${GCLOUD_KEY}

docker tag ${CONTAINER_IMAGE_NAME}:${APP_ENV} gcr.io/${GCLOUD_PROJECT_ID}/${CONTAINER_IMAGE_DIR}:${APP_ENV}
echo "Tag set"

DIGEST=$(gcloud docker -- push gcr.io/${GCLOUD_PROJECT_ID}/${CONTAINER_IMAGE_DIR}:${APP_ENV} | tail -n 1 | sed -e 's/.*\(sha256.*\) size:.*/\1/')
echo "Push finished"
echo ${DIGEST}
echo "" >> deployment/.env.${APP_ENV}
echo "DIGEST=${DIGEST}" >> deployment/.env.${APP_ENV}
