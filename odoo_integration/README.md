# Odoo 18 Integration (Chat AI Bridge)

This directory provides an Odoo 18 module that connects your Odoo instance to the Chat AI backend in this repository.

## What it does

- Adds a **Chat AI** menu item in Odoo.
- Provides a simple UI to send a prompt and receive JSON results.
- Uses Odoo settings to store the Chat AI API base URL, optional API key, and model.

## Best deployment scenarios

- **External AI service (recommended):** Run this backend separately (Docker/Kubernetes/VM) and let Odoo call it via HTTPS. This keeps heavy AI workloads out of the Odoo worker, scales independently, and simplifies model upgrades.
- **Internal service (embedded in Odoo):** Only recommended for small deployments with strict data residency where the AI model must run on the same host and the workload is light. This increases resource pressure on Odoo workers and complicates upgrades.

## Embedded vs. external integration

- **External integration (this module):** Odoo stores connection settings and proxies user prompts to the AI backend. Best for scale and reliability.
- **Embedded module:** You would bundle model/runtime inside Odoo (custom Python deps, GPU drivers, etc.). It is possible but requires more maintenance and tight coupling.

## Install

1. Copy the module folder into your Odoo addons path:

```bash
cp -R odoo_integration/chat_ai_bridge /path/to/odoo/addons/
```

2. Restart Odoo, update the app list, and install **Chat AI Bridge**.

## Configure

In **Settings → Chat AI**, configure:

- **Chat AI API Base URL** (e.g., `https://your-ai-service.example.com`)
- **Chat AI API Key** (optional)
- **Default Model** (optional, e.g., `openai:gpt-4o-mini`)
- **Smart SQL Generation** (toggle)

## Use

Open **Chat AI** from the main menu. Enter a prompt and press **Send**. The response is shown as formatted JSON.

## Backend endpoint

This module expects the backend to expose:

- `POST /api/generate-sql`
- `POST /api/smart-generate-sql`

See `ChatWithDB_Backend/README.md` for backend setup.
