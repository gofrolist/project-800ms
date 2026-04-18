/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  // Tenant API key baked into the SPA. Treat as public — anyone who opens
  // the page can read it. Use a tenant with origin + rate-limit scoped
  // tightly to the SPA's domain. Full auth flows belong on a real backend.
  readonly VITE_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
