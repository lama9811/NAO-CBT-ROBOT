// Shared helpers for the hosted NAO dashboard.
import { Redis } from "@upstash/redis";
import crypto from "node:crypto";

export const SNAPSHOT_KEY = "nao:snapshot";

// Find the Upstash REST credentials whatever prefix Vercel gave them.
// Connecting Upstash from Vercel's Storage tab lets you pick a prefix: the
// default is KV_ (KV_REST_API_URL / KV_REST_API_TOKEN), a direct Upstash
// setup uses UPSTASH_REDIS_REST_URL / _TOKEN, and a custom prefix gives
// <PREFIX>_REST_API_URL / _TOKEN or <PREFIX>_KV_REST_API_URL / _TOKEN.
export function findRedisEnv(env = process.env) {
  const pairs = [
    ["UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN"],
    ["KV_REST_API_URL", "KV_REST_API_TOKEN"],
  ];
  for (const [u, t] of pairs) if (env[u] && env[t]) return { url: env[u], token: env[t] };
  for (const key of Object.keys(env)) {
    const m = key.match(/^(.*)REST_API_URL$/) || key.match(/^(.*)REDIS_REST_URL$/);
    if (!m) continue;
    const tokenKey = key.replace(/URL$/, "TOKEN");
    if (env[tokenKey] && !/READ_ONLY/.test(tokenKey)) return { url: env[key], token: env[tokenKey] };
  }
  return null;
}

export function redis() {
  const found = findRedisEnv();
  return found ? new Redis(found) : null;
}

export function safeEqual(a, b) {
  const x = Buffer.from(String(a));
  const y = Buffer.from(String(b));
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}
