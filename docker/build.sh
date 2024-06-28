#!/bin/bash

set -ex

# we set the docker name to the project name and use the tag to indicate
# the version/env 
: "${PROJECT_NAME:=case_study}"
: "${DOCKER_TAG:=v1.0}"

# # # # # # # # # # #
# BUILD
# # # # # # # # # # #

# actual build
docker build \
  --build-arg HOST_UNAME=$USER \
  --build-arg HOST_UID=$(id -u) \
  --build-arg HOST_GID=$(id -g) \
  -t $PROJECT_NAME:$DOCKER_TAG \
  -f docker/Dockerfile .