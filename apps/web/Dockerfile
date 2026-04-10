FROM oven/bun:1.3.9-alpine AS build

WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY . .
ARG VITE_API_URL
RUN bun run build

FROM busybox:1.37-musl AS dist
COPY --from=build /app/dist /dist
CMD ["true"]
