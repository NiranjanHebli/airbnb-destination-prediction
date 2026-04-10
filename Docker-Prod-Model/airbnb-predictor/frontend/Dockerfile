# ── Stage 1: Build ──────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app
COPY package.json .
RUN npm install

COPY public/ ./public/
COPY src/ ./src/

ARG REACT_APP_BACKEND_URL=http://localhost:4000
ENV REACT_APP_BACKEND_URL=$REACT_APP_BACKEND_URL

RUN npm run build

# ── Stage 2: Serve with nginx ────────────────────────────────────
FROM nginx:stable-alpine

COPY --from=builder /app/build /usr/share/nginx/html

# nginx config to handle React Router (SPA fallback)
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location / {\n\
        try_files $uri /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
