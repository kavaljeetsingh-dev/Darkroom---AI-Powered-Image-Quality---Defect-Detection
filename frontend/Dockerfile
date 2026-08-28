FROM node:22-slim AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci --silent
COPY . .
# Baked in at build time - the browser calls this URL directly, so in a
# docker-compose setup it must be reachable from the *user's browser*
# (e.g. http://localhost:8000), not just from inside the compose network.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q -O- http://localhost:80 || exit 1
CMD ["nginx", "-g", "daemon off;"]
