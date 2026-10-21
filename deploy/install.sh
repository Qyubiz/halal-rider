#!/usr/bin/env bash
# Halal rider installer for Ubuntu (AWS Lightsail). Run with:
#   curl -sL https://raw.githubusercontent.com/Qyubiz/halal-rider/main/deploy/install.sh | sudo bash
set -euo pipefail
REPO="https://github.com/Qyubiz/halal-rider"
APP=/opt/halal-rider/app
DATA=/opt/halal-rider/data

apt-get update -y -qq
apt-get install -y -qq python3 git >/dev/null
id rider >/dev/null 2>&1 || useradd --system --home-dir /opt/halal-rider --shell /usr/sbin/nologin rider
mkdir -p /opt/halal-rider "$DATA"
if [ -d "$APP/.git" ]; then git -C "$APP" pull -q; else git clone -q "$REPO" "$APP"; fi
chown -R rider:rider /opt/halal-rider
install -m 755 "$APP/deploy/halal-rider-setup" /usr/local/bin/halal-rider-setup
install -m 755 "$APP/deploy/halal-rider" /usr/local/bin/halal-rider
install -m 644 "$APP/deploy/halal-rider.service" /etc/systemd/system/halal-rider.service
systemctl daemon-reload
if systemctl is-active --quiet halal-rider; then systemctl restart halal-rider; echo "Updated and restarted."; fi
echo ""
echo "Installed. This server's public IP (add it to your Binance API key): $(curl -s https://checkip.amazonaws.com || echo unknown)"
echo "Next: sudo halal-rider-setup"
