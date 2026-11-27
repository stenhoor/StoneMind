// Serve the app on http://localhost:8080 for local development (PORT to change it).
import { serve } from './server.mjs';

const { url } = await serve({ port: Number(process.env.PORT ?? 8080) });
console.log(`StoneMind app: ${url}`);
