# CLAUDE.md

## Final goal
This repo powers a fully autonomous Shorts system: a Scout agent picks niches and topics
and directs a Maker agent that creates and uploads 2 Shorts per day. The owner should not
need to touch anything. Every change must move toward full automation: no manual steps,
no interactive prompts, self-recovery from errors, and clear logs/alerts only when
something truly needs a human.

## Key rule: secrets (never break this)
- NEVER commit secrets: API keys, YouTube/OAuth tokens, the Studio token, passwords,
  cookies, or credential files.
- .env and all credential files (client_secret*.json, token*.json, *.pem, *.key)
  must be in .gitignore.
- Secrets come only from environment variables at runtime (.env locally, GitHub Actions
  secrets or the server's env in production). Never hardcode them, not even in tests,
  examples, comments or logs.
- .env.example contains variable names and placeholders only, never real values.
- Never print secrets to logs or console output.
- Before every commit, run gitleaks. If it finds anything, do not commit; fix it first.
- If a secret is ever committed by accident: stop, tell the owner immediately, and
  treat the key as burned (it must be revoked, not just deleted).

## Other rules
- This repo is a public fork. Private business logic (Scout strategy, niche research,
  channel config) goes in a separate private repo, not here.
- Never run code from untrusted sources without a scan.
- Keep dependencies pinned with hashes.
