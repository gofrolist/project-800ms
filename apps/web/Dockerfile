FROM oven/bun:1.3.9-alpine AS build

WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY . .
# Bake a placeholder by default so one published image works for any domain.
# The final stage rewrites it at container start using the API_URL env var.
# Local builds can still pass a real URL via --build-arg to skip the runtime
# substitution (the sed below becomes a no-op when the placeholder isn't
# present).
ARG VITE_API_URL=__API_URL__
ENV VITE_API_URL=$VITE_API_URL
RUN bun run build

FROM busybox:1.37-musl AS dist
COPY --from=build /app/dist /dist-template
# Runtime substitution: copy template into the shared volume mounted at
# /dist, then rewrite any __API_URL__ occurrences. Caddy serves /dist via
# the web_dist volume after this one-shot container exits.
CMD ["/bin/sh", "-c", "set -e; \
: \"${API_URL:?API_URL env var required (e.g. https://api.example.com)}\"; \
cp -r /dist-template/. /dist/; \
grep -rl __API_URL__ /dist 2>/dev/null | xargs -r sed -i \"s|__API_URL__|${API_URL}|g\"; \
echo \"web dist staged for ${API_URL}\""]
