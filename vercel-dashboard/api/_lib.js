// Shared helpers for the hosted NAO dashboard.
import { Redis } from "@upstash/redis";
import crypto from "node:crypto";

export const SNAPSHOT_KEY = "nao:snapshot";
const COOKIE = "nao_dash";

// Vercel's Upstash integration has used two naming schemes; accept both.
export function redis() {
  const url = process.env.UPSTASH_REDIS_REST_URL || process.env.KV_REST_API_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN || process.env.KV_REST_API_TOKEN;
  if (!url || !token) return null;
  return new Redis({ url, token });
}

export function credentials() {
  return {
    user: (process.env.DASHBOARD_USER || "").trim(),
    pw: (process.env.DASHBOARD_PASSWORD || "").trim(),
  };
}

export function configured() {
  const { user, pw } = credentials();
  return Boolean(user && pw);
}

function token(user, pw) {
  return crypto.createHmac("sha256", pw).update("nao-dashboard-v2:" + user).digest("hex");
}

export function safeEqual(a, b) {
  const x = Buffer.from(String(a));
  const y = Buffer.from(String(b));
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}

export function authed(req) {
  if (!configured()) return false;
  const { user, pw } = credentials();
  const raw = req.headers.cookie || "";
  const m = raw.split(/;\s*/).find((c) => c.startsWith(COOKIE + "="));
  return Boolean(m) && safeEqual(m.slice(COOKIE.length + 1), token(user, pw));
}

export function sessionCookie(clear = false) {
  const { user, pw } = credentials();
  const value = clear ? "" : token(user, pw);
  const age = clear ? 0 : 60 * 60 * 24 * 30;
  return `${COOKIE}=${value}; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=${age}`;
}
