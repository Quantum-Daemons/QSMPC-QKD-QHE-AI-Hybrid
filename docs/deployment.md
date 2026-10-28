# Deployment runbook

The canonical deploy procedure for the public demo. Before Round 22 this existed only inside two
gitignored documents, restated in two languages, so the commands drifted between them and none of it
would have survived losing them. This file is now the single source.

Three things below are not obvious, were each learned by breaking a live deploy, and are the reason
this file exists rather than a README paragraph: `!reset` on the ports list, the shared-Caddy
network, and clone-verify before touching the live tree.

## Run it, do not follow it: `scripts/deploy_demo.sh`

The sequence below is also a script, and the script is the preferred form because one step of it has
been skipped in **six consecutive rounds**. `DP7` requires the box to be returned to `main` once the
round's PR merges; that moment is always after the session which ran the checklist has ended, so the
only person who could discharge it was never present when it came due. Round 39 rewrote the row's
wording and left that structure, and it failed a sixth time.

    bash scripts/deploy_demo.sh <branch>          # deploy, then verify the PUBLIC url
    bash scripts/deploy_demo.sh --return-to-main  # after the merge: this is what discharges DP7

It asserts rather than assumes: the build's exit status is captured, the container's networks are
checked against the shared-proxy requirement (the 502 trap below), the public URL must answer 200,
and `--return-to-main` re-reads the branch afterwards rather than trusting the checkout. Connection
details come from `QSH_DEMO_SSH`, `QSH_DEMO_KEY`, `QSH_DEMO_DIR` and `QSH_DEMO_URL` in the
environment - the script is tracked, so it carries none of them.

The steps below remain the reference for what the script does and why, and for what it does not cover
(the clone-verify step is still a manual judgement call).

## The rule: deploy from git, never by copying

Commit, push, then have the server fetch. Copying a working directory to the server ships untracked
files and silently omits directories that are present locally but never committed - which has
happened here, and the failure appears at runtime as a missing module, long after the copy reported
success.

    # 1. local: commit and push
    git add -A && git commit && git push -u origin <branch>

    # 2. server: prove the commit stands alone BEFORE touching the live tree
    cd ~/dev && git clone --branch <branch> --depth 1 <repo> clone-verify
    cd clone-verify && ls web/src/components/ src/qsh/models/    # anything missing = incomplete commit

    # 3. server: reset the deploy tree to exactly that commit
    cd ~/dev/QSMPC-QKD-QHE-AI-Hybrid
    git fetch origin && git reset --hard origin/<branch>
    git rev-parse --short HEAD                                   # record this; it is the deployed version

Step 2 is the one worth keeping. A fresh clone of the pushed commit is the only check that answers
"does this commit contain everything the build needs", and it costs seconds.

## Demo tier, standalone (this host owns 80 and 443)

    docker compose -p qsh-demo -f docker/docker-compose.demo.yml build web
    SITE_ADDRESS=your-domain.example \
      docker compose -p qsh-demo -f docker/docker-compose.demo.yml up -d web
    docker compose -p qsh-demo -f docker/docker-compose.demo.yml logs -f web

`SITE_ADDRESS` sets the Caddy site address. A bare domain gets automatic Let's Encrypt TLS, which
needs the DNS A record already pointing at the host and both 80 and 443 reachable - 80 is used for
the challenge even though the site ends up on 443.

## Demo tier, behind a reverse proxy that already owns 80 and 443

This is how the public demo runs: another project's Caddy already holds the public ports.

    docker compose -p qsh-demo \
      -f docker/docker-compose.demo.yml \
      -f docker/docker-compose.demo.behind-proxy.yml \
      build web && \
    docker compose -p qsh-demo \
      -f docker/docker-compose.demo.yml \
      -f docker/docker-compose.demo.behind-proxy.yml \
      up -d web

**Both `-p qsh-demo` and the override are required.** Drop either and the container starts, `docker
ps` reports `Up`, and the site returns 502 - because the container is not on the network the front
proxy resolves names over. Always confirm the network after starting:

    docker inspect -f '{{range $k,$_ := .NetworkSettings.Networks}}{{$k}} {{end}}' qsh-demo-web-1

The container must appear on the front proxy's network, not only on its own compose default.

**Why the override uses `!reset` on the ports list.** The override must publish no host ports, and a
bare `ports: []` does not do that: Compose *appends* override ports to the base list, so 80 and 443
stay bound and the deploy fails with "port is already allocated". The `!reset` tag (Compose v2.24+)
replaces the base mapping instead of merging with it.

**Front proxy site block.** Cross-origin isolation has to survive the proxy, or threaded WASM and
`SharedArrayBuffer` stop working in the browser:

    your-domain.example {
        encode zstd gzip
        reverse_proxy qsh-demo-web-1:80
        header {
            Cross-Origin-Opener-Policy   "same-origin"
            Cross-Origin-Embedder-Policy "require-corp"
            Cross-Origin-Resource-Policy "cross-origin"
        }
    }

These headers must be served on the top-level navigation (`index.html`), not only on the `.wasm`
responses. Apply front-proxy changes with a restart, not a reload: a reload misses edits to a
bind-mounted Caddyfile.

## Full tier (server-side AI, MCP, metrics)

    docker compose -f docker/docker-compose.yml build
    docker compose -f docker/docker-compose.yml up -d
    docker compose -f docker/docker-compose.yml ps

    #  SPA :8080  -  API :8000/api/health  -  Grafana :3000  -  Prometheus :9090

    # end-to-end federated run and the real MCP server live in this image:
    docker compose -f docker/docker-compose.yml exec api \
      python -m qsh.orchestration.pipeline --usecase medical_fl --out results/

## Verify the deploy, in this order

Order matters. The first command says what the server sends; it says nothing about what a browser
tab is running.

    # 1. which bundle does the server serve, and does it contain the change?
    curl -s https://your-domain.example/ | grep -o '/assets/index-[A-Za-z0-9_-]*\.js'
    curl -s https://your-domain.example/assets/index-<hash>.js | grep -c "<a string literal you added>"

Grep for a **string literal**. Function names are minified, so only literals survive the build.

    # 2. then cache-bust the tab and confirm which bundle it actually loaded
    #    open https://your-domain.example/?cb=<timestamp>
    document.querySelectorAll('script[src]')

    # 3. in the browser, at that origin:
    crossOriginIsolated        // -> true
    typeof SharedArrayBuffer   // -> "function"

Skipping step 2 has produced a false negative here: after a deploy, navigating to the same URL
re-served the cached bundle, so a working fix looked broken. The failure is asymmetric - a stale tab
makes a good fix look bad, you change it again, and the second change is the one that ships
unverified.

Full browser acceptance criteria live in the project's browser verification checklist, which is maintained privately and is not part of this repository - it carries the deploy procedure and the internal failure history. The observable outcomes it gates are recorded in `verification-log.md`, which IS published.

## When TLS or the proxy misbehaves

    docker logs caddy --since 10m 2>&1 | grep -iE 'certificate|obtain|acme|challenge|error|rate'
    curl -vI https://your-domain.example 2>&1 | head -20    # 'curl -s' HIDES the real TLS error
    docker exec caddy wget -qO- http://qsh-demo-web-1/ | head -c 80   # upstream reachable?

Note the middle line: `-s` suppresses exactly the error you are looking for.

## Host hardening

Applied once, at provisioning: a non-root sudo user inheriting the root `authorized_keys`; SSH
key-only with root login disabled; `ufw` denying inbound except SSH, HTTP and HTTPS (HTTP is required
for the ACME challenge); `fail2ban`; unattended security upgrades; Docker Engine with the compose
plugin from the official script.

---

> Moved here from the README in Round 80, by the user's request to keep the README to a summary; the text is unchanged apart from its link paths, and the README links here.

## Deployment tiers, from the README

Host-specific deployment - the port-collision workaround, the shared-proxy override and the
public-demo profile - is in [`docs/deployment.md`](deployment.md). That file had **no inbound
link from any tracked file** until Round 43, and it is the only document carrying the workaround, so
a cloner hitting the collision had no way to find the answer that already existed.

A generated 14-section technical report is at
[`report/technical-report.pdf`](../report/technical-report.pdf); build it with `make report`. Every
table in it is generated from the repository's own artefacts rather than transcribed.

- **Full deployment (federated learning + inference)** - Docker image build -> server startup:
  ```bash
  git clone <repo> qsh && cd qsh
  docker compose -f docker/docker-compose.yml build       # build the images (full AI tier + OpenFHE + Node/MCP)
  docker images | grep -E 'docker-(api|web)'           # the freshly-built images
  docker compose -f docker/docker-compose.yml up -d       # start: API (+AI tier) . SPA . Prometheus . Grafana
  docker compose -f docker/docker-compose.yml ps         # all services healthy
  # SPA http://localhost:8080  .  API http://localhost:8000/api/health  .  Grafana http://localhost:3000
  # run a federated use case end-to-end (writes results/):
  docker compose -f docker/docker-compose.yml exec api \
    python -m qsh.orchestration.pipeline --usecase medical_fl --out results/
  ```
  The full `Dockerfile.api` (Debian-trixie) **bakes in the AI tier** (CPU torch + PennyLane + MedMNIST),
  the **live OpenFHE** CKKS backend, and **Node/npx** for the real MCP server, so a federated `medical_fl` run +
  the MCP server work out of the box. Fits a **6-core / 12 GB / 100 GB** box (CPU; GPU passthrough via host
  `nvidia-container-toolkit` + the `deploy.resources` GPU block is optional and faster). Ports are env-overridable
  (`WEB_PORT` / `API_PORT` / `GRAFANA_PORT`).
- **Client-side build (implementation capability)**: the SPA also builds **fully client-side** - every
  computation runs in the browser (real CKKS FHE + a Polytune garbled-circuit 2PC, ONNX inference, a
  simulated BB84 channel, PSI / SHAP / QFL), with **zero `/api` compute**. In this mode the system needs **neither the backend
  FHE/MPC compute path nor the frontend FHE/MPC backend-selectors** - the client owns both - so it requires
  no GPU and no server-side ML. The same pages and buttons work in both modes.
- **Monitoring**: see [`monitoring/README.md`](../monitoring/README.md) (Prometheus + Grafana for the local
  full deployment; Zabbix dropped as redundant).
