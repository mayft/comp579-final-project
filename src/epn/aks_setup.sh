#!/bin/bash
RESOURCE_GROUP=""
CLUSTER_NAME=""
LOCATION="" 

# Create the Resource Group
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "Creating AKS Cluster with Dedicated Head Node Pool..."
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --node-count 1 \
  --node-vm-size Standard_NC12 \
  --nodepool-name dedicated \
  --nodepool-labels role=headnode \
  --generate-ssh-keys

echo "Adding Spot Node Pool for Workers..."
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name ondemandworkers \
  --node-vm-size Standard_DS2_v2 \
  --node-count 18 \
  --labels role=worker

az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME

echo "Installing KubeRay Operator..."
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator \
  --version 1.0.0 \
  --create-namespace \
  --namespace ray-system

echo "AKS Setup and KubeRay Installation Complete!"