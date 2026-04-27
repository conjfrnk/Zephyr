#!/bin/bash
set -euo pipefail

REGION=${AWS_REGION:-us-east-1}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/zephyr"
STACK_NAME="zephyr-app"

# Login to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO

# Build and push (Fargate default is linux/amd64; pin platform so Apple Silicon
# hosts produce a deployable image)
docker buildx build --platform linux/amd64 --provenance=false \
  -t ${ECR_REPO}:latest --push .

# Deploy/update CloudFormation stack
aws cloudformation deploy \
  --template-file cloudformation.yml \
  --stack-name $STACK_NAME \
  --parameter-overrides \
    DockerImage=${ECR_REPO}:latest \
    DomainName=zephyrmap.com \
  --capabilities CAPABILITY_IAM \
  --region $REGION

# Force new deployment to pick up latest image
CLUSTER=$(aws cloudformation describe-stack-resource --stack-name $STACK_NAME --logical-resource-id ECSCluster --query 'StackResourceDetail.PhysicalResourceId' --output text --region $REGION)
SERVICE=$(aws cloudformation describe-stack-resource --stack-name $STACK_NAME --logical-resource-id ECSService --query 'StackResourceDetail.PhysicalResourceId' --output text --region $REGION)
aws ecs update-service --cluster $CLUSTER --service $SERVICE --force-new-deployment --region $REGION

echo "Deployment initiated. Waiting for service stability..."
aws ecs wait services-stable --cluster $CLUSTER --services $SERVICE --region $REGION
echo "Deployment complete!"
