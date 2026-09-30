// The Pi posts its status here every 10 s. It cannot be polled from the
// cloud because it sits behind Morgan's NAT, so it reports out instead.
import { redis, safeEqual, SNAPSHOT_KEY } from "./_lib.js";

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ ok: false });
  const secret = (process.env.DASHBOARD_INGEST_SECRET || "").trim();
  const auth = String(req.headers.authorization || "");
  if (!secret || !safeEqual(auth, "Bearer " + secret)) {
    return res.status(401).json({ ok: false });
  }
  const db = redis();
  if (!db) return res.status(500).json({ ok: false, error: "no_database" });
  const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
  body.received_at = Date.now() / 1000;
  // Kept for a day so a long outage still shows the last known state.
  await db.set(SNAPSHOT_KEY, JSON.stringify(body), { ex: 60 * 60 * 24 });
  return res.status(200).json({ ok: true });
}
