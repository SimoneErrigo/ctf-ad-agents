FROM node:22-alpine AS builder
WORKDIR /app
RUN corepack enable
COPY . .
RUN pnpm install --frozen-lockfile
ARG NEXT_PUBLIC_API_URL=http://chat.ctf.local/api
ARG NEXT_PUBLIC_ASSISTANT_ID=conversational
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_PUBLIC_ASSISTANT_ID=$NEXT_PUBLIC_ASSISTANT_ID \
    NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
RUN corepack enable
COPY --from=builder /app ./
EXPOSE 3000
CMD ["pnpm", "start"]
