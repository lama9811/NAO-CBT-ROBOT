import { authed, configured, redis, SNAPSHOT_KEY } from "./_lib.js";

export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  if (!configured()) return res.status(503).json({ error: "no_password" });
  if (!authed(req)) return res.status(401).json({ error: "login" });
  const db = redis();
  if (!db) return res.status(500).json({ error: "no_database" });
  const snap = await db.get(SNAPSHOT_KEY);
  if (!snap) return res.status(200).json({ waiting: true });
  const data = typeof snap === "string" ? JSON.parse(snap) : snap;
  // How long since the Pi last reported, measured on Vercel's clock.
  data.pi_age_s = Math.max(0, Date.now() / 1000 - (data.received_at || 0));
  return res.status(200).json(data);
}
