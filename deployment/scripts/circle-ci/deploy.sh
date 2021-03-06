#!/usr/bin/env bash

set -o allexport
source ~/.bashrc
set +o allexport

set -e

if [[ "$CIRCLE_BRANCH" == "develop" ]]; then
  APP_ENV=staging
elif [[ "$CIRCLE_BRANCH" == "master" ]]; then
  APP_ENV=production
else
  APP_ENV=issue
  exit 0
fi

set -o allexport
source deployment/.env.${APP_ENV}
set +o allexport
echo 'imports done'
PATH=$PATH:/home/circleci/google-cloud-sdk/bin

gcloud auth activate-service-account ${GCLOUD_EMAIL} --key-file ${GCLOUD_KEY}
echo 'google auth done'
# Get kubectl access
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${CLOUDSDK_COMPUTE_ZONE} --project ${GCLOUD_PROJECT_ID}
echo 'google cluster accessed'
# Apply deployment changes
kubectl delete --all jobs && kubectl delete --all cronjobs
echo 'jobs deleted'
kubectl apply -f deployment/kubernetes/items/${APP_ENV}/virushack-prediction-api.yaml
kubectl apply -f deployment/kubernetes/items/${APP_ENV}/load_balancer.yaml
echo 'services deployed'
# Set image to latest revision
kubectl set image deployment ${CLUSTER_NODE} api=gcr.io/${GCLOUD_PROJECT_ID}/${CONTAINER_IMAGE_DIR}@${DIGEST} --record
echo 'image set'