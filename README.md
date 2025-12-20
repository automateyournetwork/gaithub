# gaithub
A hub for Gait repositories 

# gaithub (v0)

Minimal GAIT remote server.

Implements:
- Content-addressed object storage (canonical JSON)
- Branch refs
- Memory refs
- Missing-object negotiation

This is a localhost-first v0 server intended to back the `gait` CLI.

## Run

```bash
pip install -r requirements.txt

export GAITHUB_DATA_DIR=./data
export GAITHUB_TOKENS_JSON='{"devtoken123":"john"}'

uvicorn gaithubd.server:app --host 127.0.0.1 --port 8787
