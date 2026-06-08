#!/bin/bash


PATH=~/your/path

cd "$PATH"

NS=ctf-ad-agents

# First stop all workloads to avoid them fighting with us over the PVC
kubectl -n "$NS" scale deploy --all --replicas=0
kubectl -n "$NS" scale statefulset/postgres --replicas=0

# Wait for pod removal
kubectl -n "$NS" delete pod --all --wait=true

# Delete persistent volume claims
kubectl -n "$NS" delete pvc data-postgres-0 patcher-workspace exploiter-workspace --ignore-not-found

# Recreate resources and PVC
kubectl apply -k deploy-k8s/kustomize/infra
kubectl apply -k deploy-k8s/kustomize/mcp
kubectl apply -k deploy-k8s/kustomize/ui

# Report back workloads
helm upgrade agent-server langchain/langgraph-cloud \
  -n "$NS" \
  -f deploy-k8s/helm/agent-server/values.yaml \
  --reuse-values

kubectl -n "$NS" scale statefulset/postgres --replicas=1
kubectl -n "$NS" scale deploy --all --replicas=1

# Verify
kubectl -n "$NS" get pods,pvc