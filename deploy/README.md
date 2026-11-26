# Deploying serumgen web to a VPS

Tested against an Azure Ubuntu VM; the steps are generic enough for any
Linux VPS. This runs `serumgen web` directly under systemd — no Docker,
no reverse proxy, no domain required to get started.

## 1. Provision the VM

- Ubuntu 22.04 LTS or newer.
- **Size:** this is CPU inference on small VAE models via Python's stdlib
  `http.server` — no GPU, no heavy load. A B2s (2 vCPU / 4GB RAM) on Azure
  is comfortably enough; B1s (1GB) is risky once PyTorch is loaded.
- **Disk:** the code is tiny; budget for your own `data/` folder (presets,
  feature caches, models) on top — usually well under a few GB.

## 2. Open the port

Azure blocks all inbound traffic except SSH by default. Add an inbound
Network Security Group rule for the port you'll serve on (8765 below, or
80/443 if you later put a reverse proxy in front).

Azure Portal: your VM → Networking → Add inbound port rule → allow TCP
on the chosen port from any source (or restrict to your own IP if you
want to be stricter than the app-level password).

## 3. Install and clone

SSH into the VM, then:

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv git
git clone --recurse-submodules https://github.com/<you>/serumgen.git
cd serumgen
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`data/` is not in the repo, so bring your own. Either copy your local
library and trained models over (fastest, no retraining):

```bash
# run on your own machine
rsync -av data/library data/models <user>@<vm-ip>:~/serumgen/data/
```

or copy just your preset folder and build on the VM with
`serumgen import <folder>` followed by `serumgen train <type>` for each
type.

Keep in mind that the web UI lists the names of your library presets and
serves presets generated from them. Before putting it on a public address,
make sure you're allowed to share material derived from those presets, and
set a password.

## 4. Choose a password and configure the service

```bash
sed -i "s/YOUR_USER/$(whoami)/g; s#/home/YOUR_USER/serumgen#$(pwd)#g" \
    deploy/serumgen.service
```

Then edit `deploy/serumgen.service` and replace `change-this-password`
with a real password — this is what protects `/api/generate` and every
other route once the server is reachable from the internet (`--host
0.0.0.0` in the unit file). Don't skip this: with no password set and a
public IP, anyone who finds the URL can hit the API freely.

## 5. Install the service

```bash
sudo cp deploy/serumgen.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now serumgen
```

## 6. Verify

```bash
sudo systemctl status serumgen         # should show "active (running)"
curl -u serumgen:<your-password> http://localhost:8765/api/types
```

From your own machine: `http://<vm-public-ip>:8765/` — the browser will
prompt for the username (`serumgen`) and the password you set.

## Logs and troubleshooting

```bash
sudo journalctl -u serumgen -f         # live log tail
sudo systemctl restart serumgen        # after pulling code changes
```

## Not covered here (deliberately)

- **HTTPS / a domain name.** Optional — a reverse proxy like Caddy gives
  free auto-HTTPS with almost no config if you point a domain at the VM;
  the raw `http://<ip>:8765` URL above works fine without one for a
  quick demo.
- **Rate limiting.** Basic Auth is the main defense for a small
  friends-only audience; add rate limiting only if abuse becomes a
  real problem.
