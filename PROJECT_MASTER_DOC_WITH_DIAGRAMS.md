# Project Master Document

## Diagram — Linking via env vars + CORS

```mermaid
flowchart TD
  WEB_APP[\"Web App (Vercel)\"] -->|VITE_WS_URL (wss)| API_WS[\"Backend WS Endpoint (/ws/recognize)\"]
  WEB_APP -->|VITE_API_URL (https)| API_REST[\"Backend REST (/health)\"]

  API_WS --> CORS_CHECK[\"CORS allowlist checks Origin\"]
  API_REST --> CORS_CHECK

  CORS_CHECK --> OK_ALLOWED[\"Allowed: Vercel + localhost\"]
```

[Other content of the file remains unchanged]