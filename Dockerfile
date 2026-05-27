# AI Microservice — NestJS
# Multi-stage: builder compiles TypeScript, runner is lean production image

FROM node:20-slim AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --legacy-peer-deps

COPY tsconfig*.json ./
COPY src/ ./src/

RUN npm run build

# ---- runner ----
FROM node:20-slim AS runner

# node:20-slim already has user 'node' at uid 1000 — use it directly
WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev --legacy-peer-deps

COPY --from=builder /app/dist ./dist

RUN chown -R node:node /app
USER node

EXPOSE 3380

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD wget -qO- http://localhost:3380/health || exit 1

CMD ["node", "dist/main.js"]
