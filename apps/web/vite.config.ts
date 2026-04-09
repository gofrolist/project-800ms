import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Loopback only by default — `host: true` binds to 0.0.0.0 and exposes
    // the dev server on every interface, which is risky on a cloud VM or
    // shared LAN. Override via `vite --host` (or set host: true here)
    // when you need to test from a phone on the same network.
    host: "localhost",
  },
});
