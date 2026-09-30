import { configured, credentials, safeEqual, sessionCookie } from "./_lib.js";

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ ok: false });
  if (!configured()) return res.status(503).json({ ok: false, error: "no_password" });
  const { user, pw } = credentials();
  const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
  const okUser = safeEqual(String(body.username || "").trim().toLowerCase(), user.toLowerCase());
  const okPw = safeEqual(String(body.password || ""), pw);
  if (!(okUser && okPw)) {
    await new Promise((r) => setTimeout(r, 500)); // blunt guessing
    return res.status(401).json({ ok: false, error: "wrong_login" });
  }
  res.setHeader("Set-Cookie", sessionCookie());
  return res.status(200).json({ ok: true });
}
