#!/usr/bin/env bash
#
# Usage:
#   deploy-k8s/rollout.sh            # use the moving dev tag
#   deploy-k8s/rollout.sh <sha|tag>  # use an explicit tag, e.g. sha-1a2b3c4 or dev
#   PULL_POLICY=IfNotPresent deploy-k8s/rollout.sh  # reuse local images intentionally
set -euo pipefail

NS=ctf-ad-agents
OWNER=ghcr.io/simoneerrigo
HELM_RELEASE=agent-server
AGENT_SERVER_DEPLOY=agent-server-langgraph-cloud-api-server
HELM_CHART=langchain/langgraph-cloud
HELM_VALUES="$(cd "$(dirname "$0")" && pwd)/helm/agent-server/values.yaml"
MCP_APPS=(janus-mcp exploiter-mcp patcher-mcp)

image_exists() {
  docker manifest inspect "$1" >/dev/null 2>&1
}

# Tag to roll to: explicit arg, or the moving dev tag.
if [[ $# -ge 1 ]]; then
  TAG="$1"
else
  TAG="dev"
fi

PULL_POLICY="${PULL_POLICY:-Always}"

echo ">> Rolling ctf-ad-agents services in namespace '$NS' onto tag: $TAG"

APPS_TO_ROLL=()
for app in "${MCP_APPS[@]}"; do
  image="$OWNER/$app:$TAG"
  if image_exists "$image"; then
    APPS_TO_ROLL+=("$app")
  else
    echo ">> Skipping $app: image not found or registry not accessible: $image"
  fi
done

ROLL_AGENT_SERVER=false
if image_exists "$OWNER/agent-server:$TAG"; then
  ROLL_AGENT_SERVER=true
else
  echo ">> Skipping agent-server: image not found or registry not accessible: $OWNER/agent-server:$TAG"
fi

if [[ ${#APPS_TO_ROLL[@]} -eq 0 && "$ROLL_AGENT_SERVER" == "false" ]]; then
  echo "!! No images found for tag '$TAG'. Wait for CI to finish, pass an existing tag, or use 'dev'." >&2
  exit 1
fi

# kustomize-managed workloads: deployment name == container name for all of them.
for app in "${APPS_TO_ROLL[@]}"; do
  echo ">> kubectl set image deploy/$app -> $OWNER/$app:$TAG"
  kubectl -n "$NS" set image "deploy/$app" "$app=$OWNER/$app:$TAG"
  kubectl -n "$NS" patch "deploy/$app" --type=strategic \
    -p "{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"$app\",\"imagePullPolicy\":\"$PULL_POLICY\"}]}}}}"
  kubectl -n "$NS" rollout restart "deploy/$app"
done

# Helm-managed agent-server.
if [[ "$ROLL_AGENT_SERVER" == "true" ]]; then
  echo ">> helm upgrade $HELM_RELEASE -> $OWNER/agent-server:$TAG"
  helm upgrade "$HELM_RELEASE" "$HELM_CHART" \
    -n "$NS" \
    -f "$HELM_VALUES" \
    --set images.apiServerImage.tag="$TAG" \
    --set images.apiServerImage.pullPolicy="$PULL_POLICY" \
    --reuse-values
  kubectl -n "$NS" rollout restart "deploy/$AGENT_SERVER_DEPLOY"
fi

# Wait for the rollouts to converge.
for app in "${APPS_TO_ROLL[@]}"; do
  kubectl -n "$NS" rollout status "deploy/$app" --timeout=180s
done
if [[ "$ROLL_AGENT_SERVER" == "true" ]]; then
  kubectl -n "$NS" rollout status "deploy/$AGENT_SERVER_DEPLOY" --timeout=180s
fi

echo ">> Done. ctf-ad-agents services are on tag: $TAG"
