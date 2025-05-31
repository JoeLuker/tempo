# Multi-stage build for optimal size
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM node:20-alpine AS production

# Install serve for serving static files
RUN npm install -g serve

# Set working directory
WORKDIR /app

# Copy built application from builder stage
COPY --from=builder /app/build ./build
COPY --from=builder /app/package.json ./

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S frontend -u 1001

# Change ownership of the app directory
RUN chown -R frontend:nodejs /app
USER frontend

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/ || exit 1

# Start the application
CMD ["serve", "-s", "build", "-l", "3000"]

# Development stage
FROM node:20-alpine AS development

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev dependencies)
RUN npm ci

# Copy source code
COPY . .

# Expose port for development
EXPOSE 5173

# Start development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]