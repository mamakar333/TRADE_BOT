# Deploying to a private server

Real money and real API credentials are involved here. This runbook is
written so the dashboard is reachable only by you (password + HTTPS) and
the bot's on/off switch is never something a crash or a reboot flips back
on by itself without you noticing.

## Architecture

```
your browser --HTTPS+password--> Caddy (443) --localhost only--> Streamlit dashboard (127.0.0.1:8501)
                                                                          |
                                                                   controls, via bot_control.py
                                                                          v
                                                              run_live_trading.py (real orders)
```

- **Caddy** is the only thing listening on a public port. It terminates
  HTTPS (automatic, free certificate) and requires a login before it will
  even talk to the dashboard.
- **The dashboard** binds to `127.0.0.1` only -- it is not reachable at all
  except through Caddy, even if the firewall were somehow misconfigured.
- **The bot** is deliberately *not* systemd-managed with auto-restart. It's
  started and stopped exactly the way it already is today -- via
  `bot_control.py`, either from the dashboard's Start/Stop buttons or a
  shell. If the server reboots, the bot stays off until you explicitly
  start it again. That's intentional: a reboot silently resuming
  real-money trading without you noticing is worse than it staying off.
  The dashboard itself *does* auto-restart (systemd, `Restart=always`) --
  it's just a UI, there's no reason it shouldn't always be up so you can
  check status and hit Start yourself.
- **The watchdog** (`bot-watchdog.timer`, added 2026-08-04 after a real
  ~75min silent gap overnight with no crash or error logged) runs
  `run_watchdog.py` every 3 minutes and brings the bot back if it finds it
  unexpectedly down -- but only when the last thing you did was press
  Start, never after Stop, and never across a reboot (same "stays off"
  guarantee as above -- see `bot_control.watchdog_check()`'s docstring).
  It closes the gap between "crashed mid-session" (now auto-recovered) and
  "server rebooted" (still stays off, unchanged).

## 1. Create the server

Recommended: a small Ubuntu 24.04 droplet on DigitalOcean (~$6/mo, plenty
for this workload) -- simple UI, good docs, no card-on-file surprises
beyond the plan you pick. Any provider works the same way from here on;
substitute as you like.

1. Create a droplet: Ubuntu 24.04 LTS, cheapest tier (1GB RAM is enough).
2. When it asks for an SSH key, add your **public** key
   (`~/.ssh/id_ed25519.pub` or similar) -- never the private one. If you
   don't have one yet: `ssh-keygen -t ed25519`.
3. Note the server's public IP once it's created.
4. Confirm you can reach it: `ssh root@<ip>`.

Come back here once that works.

## 2. Provision the server

From your laptop:

```
scp deploy/provision.sh root@<ip>:/root/
ssh root@<ip> 'bash /root/provision.sh'
```

This installs Caddy, sets up the firewall (only SSH/80/443 open), enables
fail2ban, enables automatic security updates, and creates a dedicated
`tradebot` user that everything else runs as (never root).

## 3. Deploy the code

From your laptop, in this repo:

```
deploy/sync.sh tradebot@<ip>
```

This copies everything **except** secrets, databases, and logs -- those are
handled separately, next.

## 4. Transfer secrets (never through chat, never through git)

Directly, laptop to server, over the same encrypted SSH channel:

```
scp .env tradebot@<ip>:/home/tradebot/TRADE_BOT/.env
scp kalshi_prod_key.pem tradebot@<ip>:/home/tradebot/TRADE_BOT/kalshi_prod_key.pem
ssh tradebot@<ip> 'chmod 600 /home/tradebot/TRADE_BOT/.env /home/tradebot/TRADE_BOT/kalshi_prod_key.pem'
```

**Your `.env` currently has an absolute laptop path**
(`/Users/mamakar/codebase/TRADE_BOT/kalshi_prod_key.pem`), which won't exist
on the server. After copying `.env` over, fix it on the server:

```
ssh tradebot@<ip> "sed -i 's|^KALSHI_PRIVATE_KEY_PATH=.*|KALSHI_PRIVATE_KEY_PATH=/home/tradebot/TRADE_BOT/kalshi_prod_key.pem|' /home/tradebot/TRADE_BOT/.env"
```

(Your laptop's `.env` is untouched by this -- it only edits the copy on the
server.)

## 5. Install Python dependencies

```
ssh tradebot@<ip> 'cd /home/tradebot/TRADE_BOT && ~/.local/bin/uv sync'
```

## 6. Set up the dashboard as a service

```
scp deploy/dashboard.service root@<ip>:/etc/systemd/system/
ssh root@<ip> 'systemctl daemon-reload && systemctl enable --now dashboard'
```

## 6b. Set up the bot watchdog (optional but recommended)

```
scp deploy/bot-watchdog.service deploy/bot-watchdog.timer root@<ip>:/etc/systemd/system/
ssh root@<ip> 'systemctl daemon-reload && systemctl enable --now bot-watchdog.timer'
```

Brings the bot back automatically if it dies unexpectedly mid-session
(crash, an accidental kill) while you were away -- never after you press
Stop, and never across a server reboot. See the Architecture section above.

## 6c. Set up the paper-trading bot (A/B control, no real money)

No systemd unit for this one -- 2026-08-04, switched to the same manual
start/stop model as the live bot (`trade_bot/paper_bot_control.py`, a PID
file + SIGTERM, detached subprocess via `subprocess.Popen(...,
start_new_session=True)`), specifically so the Start/Stop buttons in the
dashboard and Android app actually control it. A `Restart=always` systemd
unit would fight a manual Stop the same way it would for the live bot (see
the Architecture section above) -- this one just doesn't auto-restart on
crash at all, per explicit request for a manual button rather than another
watchdog.

Start it once after `dashboard`/`api` are up, either from the dashboard's
Bot Status panel, the Android app, or directly:

```
ssh tradebot@<ip> 'curl -s -X POST http://127.0.0.1:8502/api/paper/bot/start'
```

Runs run_paper_trading.py against the same crypto market scope as the live
bot, using the ORIGINAL CryptoTechnicalStrategy configuration, unchanged --
a live A/B control for whatever strategy the real-money bot is running (see
docs/ALGORITHM.md). Simulated only: no POST method exists anywhere in this
repo for real orders, so this can never place one regardless of what the
strategy decides.

## 7. Set up Caddy (HTTPS + password)

You don't need to own a domain. The `nip.io` trick gives you a real,
trusted HTTPS certificate for your server's own IP address for free:
`<ip-with-dashes>.nip.io` (e.g. `203-0-113-5.nip.io` for `203.0.113.5`)
resolves straight to that IP, and Let's Encrypt will happily issue a
certificate for it.

Generate a strong password and its hash (run on the server, nothing sent
anywhere yet):

```
ssh root@<ip> caddy hash-password --plaintext 'PASTE_A_STRONG_PASSWORD_HERE'
```

Save that plaintext password in your password manager now -- this is the
only time it's shown. Copy the hash it prints, then fill in the template
and ship the *generated* file (not the template) to the server -- this
uses Caddy's default installed service, no systemd changes needed:

```
sed -e "s/{{DASHBOARD_HOSTNAME}}/<ip-with-dashes>.nip.io/" \
    -e "s/{{BASIC_AUTH_USER}}/<pick-a-username>/" \
    -e "s|{{BASIC_AUTH_HASH}}|<the-hash-from-the-command-above>|" \
    deploy/Caddyfile > /tmp/Caddyfile.generated

scp /tmp/Caddyfile.generated root@<ip>:/etc/caddy/Caddyfile
ssh root@<ip> 'systemctl reload caddy || systemctl restart caddy'
rm /tmp/Caddyfile.generated
```

## 8. Verify

- Visit `https://<ip-with-dashes>.nip.io` -- you should get a real padlock
  (no browser warning) and a login prompt. Log in with the username/password
  from step 7.
- You should NOT be able to reach `http://<ip>:8501` directly (connection
  refused) -- confirms the dashboard really is only reachable through Caddy.
- In the dashboard's Live Trading tab, confirm it shows STOPPED (the bot
  should not be running yet on a fresh deploy) and your real balance.
- Start it from the dashboard when you're ready, the same way you always
  have.

## Updating code later

```
deploy/sync.sh tradebot@<ip>
ssh root@<ip> 'systemctl restart dashboard'
```

If the bot is running, **stop it from the dashboard first** (check for open
positions the same way this session always has before any restart), sync,
then start it again -- exactly the same discipline as running it locally,
just over SSH instead of on your laptop.

## Emergency stop

```
ssh tradebot@<ip> 'cd TRADE_BOT && .venv/bin/python3 -c "from trade_bot import bot_control; print(bot_control.stop())"'
```

Or just hit Stop in the dashboard -- it works identically to how it does
locally.
