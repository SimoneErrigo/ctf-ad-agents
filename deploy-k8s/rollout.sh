#!/usr/bin/env bash
#
# Usage:
#   deploy-k8s/rollout.sh            # use the current git commit (HEAD)
#   deploy-k8s/rollout.sh <sha|tag>  # use an explicit tag, e.g. sha-1a2b3c4 or dev
set -euo pipefail

NS=ctf-ad-agents
OWNER=ghcr.io/simoneerrigo
HELM_RELEASE=agent-server
HELM_CHART=langchain/langgraph-cloud
HELM_VALUES="$(cd "$(dirname "$0")" && pwd)/helm/agent-server/values.yaml"

# Tag to roll to: explicit arg, or sha-<HEAD short sha>.
if [[ $# -ge 1 ]]; then
  TAG="$1"
else
  TAG="sha-$(git rev-parse --short HEAD)"
fi

echo ">> Rolling namespace '$NS' onto tag: $TAG"

# kustomize-managed workloads: deployment name == container name for all of them.
for app in janus-mcp exploiter-mcp patcher-mcp agent-chat-ui; do
  echo ">> kubectl set image deploy/$app -> $OWNER/$app:$TAG"
  kubectl -n "$NS" set image "deploy/$app" "$app=$OWNER/$app:$TAG"
done

# Helm-managed agent-server.
echo ">> helm upgrade $HELM_RELEASE -> $OWNER/agent-server:$TAG"
helm upgrade "$HELM_RELEASE" "$HELM_CHART" \
  -n "$NS" \
  -f "$HELM_VALUES" \
  --set images.apiServerImage.tag="$TAG" \
  --reuse-values

# Wait for the rollouts to converge.
for app in janus-mcp exploiter-mcp patcher-mcp agent-chat-ui; do
  kubectl -n "$NS" rollout status "deploy/$app" --timeout=180s
done

echo ">> Done. Cluster is on tag: $TAG"
