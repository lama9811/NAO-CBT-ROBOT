import { redis, SNAPSHOT_KEY } from "./_lib.js";

// Open to anyone with the link (the user's choice). Support-agent and
// crisis turns never carry words; the Pi strips them before sending.
export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  const db = redis();
  if (!db) return res.status(500).json({ error: "no_database" });
  const snap = await db.get(SNAPSHOT_KEY);
  if (!snap) return res.status(200).json({ waiting: true });
  const data = typeof snap === "string" ? JSON.parse(snap) : snap;
  // How long since the Pi last reported, measured on Vercel's clock.
  data.pi_age_s = Math.max(0, Date.now() / 1000 - (data.received_at || 0));
  return res.status(200).json(data);
}
