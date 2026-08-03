#!/usr/bin/env bash
# Run ONCE on a fresh Ubuntu 24.04 server, as root (or via sudo), before any
# code is deployed. Sets up: a dedicated non-root user, a locked-down
# firewall, brute-force protection, and Caddy (reverse proxy + automatic
# HTTPS + password gate). Safe to re-run -- every step is idempotent.
set -euo pipefail

APP_USER="tradebot"

echo "==> Updating system packages"
apt-get update -y
apt-get upgrade -y

echo "==> Installing base packages"
apt-get install -y curl git ufw fail2ban unattended-upgrades python3 python3-venv python3-pip

echo "==> Enabling unattended security upgrades"
dpkg-reconfigure -f noninteractive unattended-upgrades

echo "==> Creating dedicated non-root user '$APP_USER' (no login shell needed beyond systemd)"
if ! id -u "$APP_USER" >/dev/null 2>&1; then
    useradd --create-home --shell /bin/bash "$APP_USER"
fi

echo "==> Installing uv (Python package manager) for $APP_USER"
sudo -u "$APP_USER" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

echo "==> Installing Caddy (reverse proxy + automatic HTTPS)"
if ! command -v caddy >/dev/null 2>&1; then
    apt-get install -y debian-keyring debian-archive-keyring apt-transport-https
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -y
    apt-get install -y caddy
fi

echo "==> Configuring firewall: SSH + HTTP(S) only, everything else denied"
ufw allow OpenSSH
ufw allow 80/tcp   # required for Let's Encrypt's HTTP-01 challenge
ufw allow 443/tcp  # HTTPS -- the ONLY way the dashboard is ever reachable
ufw --force enable

echo "==> Enabling fail2ban for SSH brute-force protection"
systemctl enable --now fail2ban

echo "==> Disabling SSH password login (key-only) -- only if a key is already authorized"
if [ -s /root/.ssh/authorized_keys ] || [ -s "/home/$APP_USER/.ssh/authorized_keys" ]; then
    sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
    systemctl reload ssh || systemctl reload sshd
    echo "    Password login disabled. Make sure you can still SSH in with your key before closing this session!"
else
    echo "    Skipped: no authorized_keys found yet -- add your SSH key first, then rerun this script."
fi

echo "==> Done. Next steps:"
echo "  1. Deploy the app code to /home/$APP_USER/TRADE_BOT (see deploy/README.md)"
echo "  2. Copy .env and your Kalshi .pem key there directly via scp -- never paste them anywhere"
echo "  3. Install the systemd unit for the dashboard and configure Caddy (see deploy/README.md)"
