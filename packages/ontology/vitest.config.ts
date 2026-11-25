import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: { conditions: ["stonemind-source"] },
  test: { include: ["test/**/*.test.ts"] },
});
