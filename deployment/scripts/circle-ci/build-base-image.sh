#!/usr/bin/env bash

set -e

IMAGE_VERSION=1.2

set -o allexport
source deployment/.env.issue
set +o allexport

PATH=$PATH:/home/circleci/google-cloud-sdk/bin

gcloud auth activate-service-account ${GCLOUD_EMAIL} --key-file ${GCLOUD_KEY}

gcloud docker --authorize-only

echo "Start build base image"

docker build -t api-base:${IMAGE_VERSION} -f deployment/base-images/Dockerfile .

echo "Start tag base image"

docker tag api-base:${IMAGE_VERSION} gcr.io/${GCLOUD_PROJECT_ID}/api-base:${IMAGE_VERSION}

docker "Start push base image"

gcloud docker -- push gcr.io/${GCLOUD_PROJECT_ID}/api-base:${IMAGE_VERSION}
