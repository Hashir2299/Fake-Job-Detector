#!/bin/bash
set -e

# Use an AWS free-tier EC2 instance, for example t2.micro or t3.micro.
# Replace this image with your Docker Hub or ECR image.
IMAGE_NAME="your-dockerhub-username/fake-job-detector:latest"

yum update -y
yum install -y docker
systemctl enable docker
systemctl start docker

docker pull "$IMAGE_NAME"
docker stop fake-job-detector || true
docker rm fake-job-detector || true
docker run -d \
  --name fake-job-detector \
  -p 80:8000 \
  --restart always \
  "$IMAGE_NAME"
