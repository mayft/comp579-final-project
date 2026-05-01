#!/bin/bash
# Variables - Update these to match your Azure environment
RESOURCE_GROUP="rg-eswm-ray"
CLUSTER_NAME="aks-kuberay-cluster"
LOCATION="denmarkeast" 

# Create the Resource Group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create AKS cluster with a Dedicated GPU Node Pool (For the Head Node)
echo "Creating AKS Cluster with Dedicated Head Node Pool..."
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --node-count 1 \
  --node-vm-size Standard_NC12 \
  --nodepool-name dedicated \
  --nodepool-labels role=headnode \
  --generate-ssh-keys

# Add the Spot Node Pool (For the Worker Nodes)
echo "Adding Spot Node Pool for Workers..."
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name ondemandworkers \
  --node-vm-size Standard_DS2_v2 \
  --node-count 18 \
  --labels role=worker

# Get credentials to connect kubectl to the new cluster
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME

echo "Installing KubeRay Operator..."
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator \
  --version 1.0.0 \
  --create-namespace \
  --namespace ray-system

echo "AKS Setup and KubeRay Installation Complete!"