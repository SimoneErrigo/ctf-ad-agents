FROM python:3.12-slim AS runtime

# git + openssh-client are needed because the MCP server shells out to `git`
# for every operation, and `git push` over ssh requires the openssh client.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY patcher_mcp ./patcher_mcp

RUN pip install .

# Where service repos live inside the container. Mount your local workspace
# here at runtime (or let the container clone on first use).
RUN mkdir -p /var/lib/patcher/workspace /root/.ssh \
    && chmod 700 /root/.ssh

EXPOSE 8766

CMD ["python", "-m", "patcher_mcp"]
