/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  // Tenant API key baked into the SPA. Treat as public — anyone who opens
  // the page can read it. Use a tenant with origin + rate-limit scoped
  // tightly to the SPA's domain. Full auth flows belong on a real backend.
  readonly VITE_API_KEY?: string;
  // Comma-separated list of TTS engines visible in the engine picker.
  // Substituted at container start by the web one-shot — see
  // apps/web/Dockerfile and infra/docker-compose.tls.yml. Undefined in
  // dev (`bun run dev`) so the SPA falls back to the full catalog.
  readonly VITE_TTS_PRELOAD_ENGINES?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
