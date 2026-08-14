# AI Microservice — NestJS
# Multi-stage: builder compiles TypeScript, runner is lean production image

FROM node:20-slim AS builder

WORKDIR /app

# npm's defaults (fetch-timeout 300s, fetch-retries 2, retry-maxtimeout 60s) are too tight
# for the build host's uplink: a clean `npm ci` here takes ~259s, so a normal run finishes
# only just inside the 300s ceiling and any socket stall pushes it over. That surfaced as
# two consecutive deploy failures with npm's generic "check your proxy settings" advice and
# exit code 146 — once in this stage, once in the runner — while the registry was reachable
# the whole time. Raising the ceiling fixes the actual cause; it does not hide a failure,
# because a genuinely unreachable registry still fails, just after a longer wait.
ENV npm_config_fetch_timeout=1200000 \
    npm_config_fetch_retries=5 \
    npm_config_fetch_retry_maxtimeout=180000

COPY package*.json ./
RUN npm ci --legacy-peer-deps

COPY tsconfig*.json ./
COPY src/ ./src/

RUN npm run build

# ---- runner ----
FROM node:20-slim AS runner

RUN apt-get update   && apt-get install -y --no-install-recommends git bash ca-certificates   && rm -rf /var/lib/apt/lists/*

# node:20-slim already has user 'node' at uid 1000 — use it directly
WORKDIR /app

# Same reasoning as the builder stage — ENV does not cross a FROM boundary.
ENV npm_config_fetch_timeout=1200000 \
    npm_config_fetch_retries=5 \
    npm_config_fetch_retry_maxtimeout=180000

COPY package*.json ./
RUN npm ci --omit=dev --legacy-peer-deps

COPY --from=builder /app/dist ./dist
COPY public ./public

RUN chown -R node:node /app
USER node

EXPOSE 3380

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD wget -qO- http://localhost:3380/health || exit 1

CMD ["node", "dist/main.js"]
