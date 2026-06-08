#!/bin/bash

# variable PATH to change directory in

PATH=~/your/path

cd "$PATH"

NS=ctf-ad-agents

# Ferma tutto quello che può usare DB/PVC
kubectl -n "$NS" scale deploy --all --replicas=0
kubectl -n "$NS" scale statefulset/postgres --replicas=0

# Aspetta / forza la rimozione dei pod
kubectl -n "$NS" delete pod --all --wait=true

# Cancella la memoria persistente
kubectl -n "$NS" delete pvc data-postgres-0 patcher-workspace exploiter-workspace --ignore-not-found

# Ricrea risorse e PVC
kubectl apply -k deploy-k8s/kustomize/infra
kubectl apply -k deploy-k8s/kustomize/mcp
kubectl apply -k deploy-k8s/kustomize/ui

# Riporta su i workload
helm upgrade agent-server langchain/langgraph-cloud \
  -n "$NS" \
  -f deploy-k8s/helm/agent-server/values.yaml \
  --reuse-values

kubectl -n "$NS" scale statefulset/postgres --replicas=1
kubectl -n "$NS" scale deploy --all --replicas=1

# Verifica
kubectl -n "$NS" get pods,pvc