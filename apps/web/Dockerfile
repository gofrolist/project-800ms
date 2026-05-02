FROM oven/bun:1.3.9-alpine AS build

WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY . .
# Bake placeholders by default so one published image works for any
# deployment. The final stage rewrites them at container start using the
# API_URL + API_KEY env vars. Local builds can still pass real values via
# --build-arg to skip the runtime substitution (sed becomes a no-op when
# the placeholder isn't present).
ARG VITE_API_URL=__API_URL__
ARG VITE_API_KEY=__API_KEY__
ARG VITE_TTS_PRELOAD_ENGINES=__TTS_PRELOAD_ENGINES__
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_API_KEY=$VITE_API_KEY
ENV VITE_TTS_PRELOAD_ENGINES=$VITE_TTS_PRELOAD_ENGINES
RUN bun run build

FROM busybox:1.37-musl AS dist
COPY --from=build /app/dist /dist-template
# Runtime substitution: copy template into the shared volume mounted at
# /dist, then rewrite __API_URL__ and __API_KEY__ occurrences. Caddy
# serves /dist via the web_dist volume after this one-shot container exits.
CMD ["/bin/sh", "-c", "set -e; \
: \"${API_URL:?API_URL env var required (e.g. https://api.example.com)}\"; \
: \"${API_KEY:?API_KEY env var required (tenant X-API-Key baked into the SPA)}\"; \
: \"${TTS_PRELOAD_ENGINES:?TTS_PRELOAD_ENGINES env var required (comma-separated subset of: piper,silero,qwen3,xtts)}\"; \
cp -r /dist-template/. /dist/; \
grep -rl __API_URL__ /dist 2>/dev/null | xargs -r sed -i \"s|__API_URL__|${API_URL}|g\"; \
grep -rl __API_KEY__ /dist 2>/dev/null | xargs -r sed -i \"s|__API_KEY__|${API_KEY}|g\"; \
grep -rl __TTS_PRELOAD_ENGINES__ /dist 2>/dev/null | xargs -r sed -i \"s|__TTS_PRELOAD_ENGINES__|${TTS_PRELOAD_ENGINES}|g\"; \
echo \"web dist staged for ${API_URL} engines=${TTS_PRELOAD_ENGINES}\""]
