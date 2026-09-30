import { sessionCookie } from "./_lib.js";

export default function handler(req, res) {
  res.setHeader("Set-Cookie", sessionCookie(true));
  return res.status(200).json({ ok: true });
}
