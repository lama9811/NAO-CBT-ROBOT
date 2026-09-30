// Shared helpers for the hosted NAO dashboard.
import { Redis } from "@upstash/redis";
import crypto from "node:crypto";

export const SNAPSHOT_KEY = "nao:snapshot";

// Vercel's Upstash integration has used two naming schemes; accept both.
export function redis() {
  const url = process.env.UPSTASH_REDIS_REST_URL || process.env.KV_REST_API_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN || process.env.KV_REST_API_TOKEN;
  if (!url || !token) return null;
  return new Redis({ url, token });
}

export function safeEqual(a, b) {
  const x = Buffer.from(String(a));
  const y = Buffer.from(String(b));
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}
