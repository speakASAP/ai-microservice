FROM node:18-alpine

# Create non-root user first
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Set working directory
WORKDIR /app

# Copy package files with correct ownership
COPY --chown=nextjs:nodejs package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code with correct ownership
COPY --chown=nextjs:nodejs . .

# Build the application
RUN npm run build

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${FRONTEND_PORT:-3000}/api/health || exit 1

# Start the application
CMD ["npm", "start"]
