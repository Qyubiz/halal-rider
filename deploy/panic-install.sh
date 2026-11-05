#!/usr/bin/env bash
# Panic buyer installer for Ubuntu (it can share the halal-rider server). Run with:
#   curl -sL https://raw.githubusercontent.com/Qyubiz/halal-rider/main/deploy/panic-install.sh | sudo bash
set -euo pipefail
REPO="https://github.com/Qyubiz/halal-rider"
APP=/opt/panic-buyer/app
DATA=/opt/panic-buyer/data

apt-get update -y -qq
apt-get install -y -qq python3 git >/dev/null
id panicbot >/dev/null 2>&1 || useradd --system --home-dir /opt/panic-buyer --shell /usr/sbin/nologin panicbot
mkdir -p /opt/panic-buyer "$DATA"
if [ -d "$APP/.git" ]; then git -c safe.directory="$APP" -C "$APP" pull -q; else git clone -q "$REPO" "$APP"; fi
chown -R panicbot:panicbot /opt/panic-buyer
install -m 755 "$APP/deploy/panic-buyer-setup" /usr/local/bin/panic-buyer-setup
install -m 755 "$APP/deploy/panic-buyer" /usr/local/bin/panic-buyer
install -m 644 "$APP/deploy/panic-buyer.service" /etc/systemd/system/panic-buyer.service
install -m 644 "$APP/deploy/panic-paper@.service" /etc/systemd/system/panic-paper@.service
systemctl daemon-reload
for n in normal 90wr ensemble; do systemctl is-active --quiet "panic-paper@$n" && systemctl restart "panic-paper@$n"; done || true
if systemctl is-active --quiet panic-buyer; then systemctl restart panic-buyer; echo "Updated and restarted."; fi
echo ""
echo "Installed. This server's public IP (must be allowed on your Binance API key): $(curl -s https://checkip.amazonaws.com || echo unknown)"
echo "Next: sudo panic-buyer-setup"
