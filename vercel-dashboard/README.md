# NAO dashboard on Vercel

A hosted copy of the Pi's `/dashboard`. The Pi is on Morgan's private
network, so this site never contacts it; the Pi posts its status to
`/api/ingest` every 10 seconds and this site stores the latest copy in
Upstash Redis.

`index.html` and `nao.jpg` are copies of `server/dashboard_static/`; a test
fails if they drift. Edit the originals and copy them here.

## Setup

1. vercel.com: Add New, Project, import this repo, set Root Directory to
   `vercel-dashboard`, deploy.
2. Project: Storage, Upstash Redis, connect (free plan).
3. Project: Settings, Environment Variables: `DASHBOARD_INGEST_SECRET`
   (the same value as the Pi's). Redeploy.
4. Pi `.env`: `DASHBOARD_REMOTE_URL=https://<your-site>.vercel.app` and the
   same `DASHBOARD_INGEST_SECRET`, then restart nao-server.

There is no login: anyone with the link can view the page. Support-agent
and crisis turns are never sent. By default the Pi sends status only (no
words at all); set `DASHBOARD_PUSH_CONVERSATION=1` on the Pi to include
non-support questions and answers.
