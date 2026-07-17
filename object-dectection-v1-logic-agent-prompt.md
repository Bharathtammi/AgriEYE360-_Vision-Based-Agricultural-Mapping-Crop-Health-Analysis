Integrate my Roboflow Workflow "object dectection v1 Logic" into this codebase.

You're connected to Roboflow through its MCP server (OAuth), so call the Roboflow tools directly and ground everything below in the workflow's real definition instead of guessing.

## Workflow
- Name: object dectection v1 Logic
- workspace_name (workspace slug): bharaths-workspace-3fhlw
- workflow_id (workflow slug, NOT the document id): object-dectection-v1-logic
- Run endpoint (POST): https://serverless.roboflow.com/bharaths-workspace-3fhlw/workflows/object-dectection-v1-logic
- Declared inputs: image (image)

## 1. Ground the integration first (before writing code)
- Call the Roboflow MCP `workflows_get` for workflow_id "object-dectection-v1-logic" and read its exact inputs, parameters, and outputs — treat that as the source of truth.
- Call `workflows_run` once with a representative image to capture a real example response. The result is a list (one entry per input image); each entry is a dict keyed by the workflow's own output names. Model your types/parsing on that real response — do not hard-code output names.

## 2. Build the client — fit this repo's language and conventions
- Detect the language/framework already used here and integrate idiomatically, reusing the repo's existing HTTP, config, and secret-handling patterns. Don't impose a new toolchain or env manager.
- Pick the client accordingly:
    - **Python** → the official `inference-sdk` package (`InferenceHTTPClient`); call `run_workflow(workspace_name="bharaths-workspace-3fhlw", workflow_id="object-dectection-v1-logic", images=..., parameters=...)` with `api_url="https://serverless.roboflow.com"`.
    - **JavaScript / TypeScript** → the Roboflow JS client (github.com/roboflow/inference-sdk).
    - **Any other language** → call the REST endpoint directly and model the client on those SDKs:
      ```
      POST https://serverless.roboflow.com/bharaths-workspace-3fhlw/workflows/object-dectection-v1-logic
      Content-Type: application/json
      { "api_key": "<from env>", "inputs": { "image": { "type": "url", "value": "https://..." } } }
      ```
      (URL inputs must be https — plain http:// is rejected; base64 inputs use `{ "type": "base64", "value": "..." }`.)
- Pass `parameters` that match the workflow's declared parameters (names + types) exactly.
- Load your Roboflow API key from an environment variable — never hardcode it (find it at app.roboflow.com/settings/api).
- Wrap inference in one well-named function with a request timeout and a couple of retries with backoff; raise clear, typed errors on failure.

## 3. Handle responses safely
- Parse defensively from the real output keys, not assumptions.
- Image-shaped outputs come back as base64 blobs (often hundreds of KB) — decode and write them to disk; never log them or hold them in memory longer than needed.
- Keep payloads small: only read the fields you use; for segmentation, don't carry raw polygon `points`.

## 4. Verify and document
- Add a smoke test that runs the function on one sample image and asserts the expected output keys exist; run it and iterate until it passes.
- Update the project's existing docs and developer instructions to cover this integration — README usage/setup, any onboarding/dev docs, and any coding-agent instruction files already in the repo (e.g. AGENTS.md, CLAUDE.md, .cursor/rules, .github/copilot-instructions.md). Only update files that already exist — don't create these from scratch.

## Guardrails
- Stop and ask before installing heavy/native dependencies, changing build or CI config, or editing unrelated files.
- Keep the change focused on this integration, and don't commit secrets.
- If this actually needs to run on live video (webcam/RTSP/file) instead of images, pause and ask me — Roboflow uses a different (WebRTC) path for that.