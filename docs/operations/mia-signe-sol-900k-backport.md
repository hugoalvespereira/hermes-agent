# Mia / Signe: bounded Sol 900K backport

Base: `fcbd1076a93841fa88855acce810e342a5b78101` (existing deployed runtime).
Reference implementation: `ade4814466f9b535d0cb549da893d3517b6f9b2b`,
`agent/model_metadata.py` context-variant helpers, max-context guard, and Codex wire normalization.

Scope: only `gpt-6-sol-900k` and its dated Sol snapshots, metadata resolution,
main Responses transport and auxiliary Responses requests. The configured model
identifier remains exact; the wire identifier is `gpt-6-sol`. Context resolves
900000 only for the explicit alias and stale 272000 advertisement; an explicit
catalog maximum caps the bump and other advertised values remain authoritative.
Existing 5.x behavior is intentionally retained. No model picker/catalog overhaul,
Mesh adaptation, package upgrade, auth migration, or memory repair.

Target settings: `model.default: gpt-6-sol-900k`, `model.provider: openai-codex`,
`agent.reasoning_effort: medium`. Existing session overrides are not erased.

Focused tests (canonical hermetic runner): 17 new contracts plus 115 existing
adapter/transport tests passed. Real inference and service receipts are captured
on each target under `~/.hermes/backups/sol-900k-rollout/`.

Deployment: fast-forward exact backport commit while leaving existing uncommitted
Mesh patches in place. Back up config and hash protected Mesh/persona files before
activation. Restart only previously active services (Mia gateway + dashboard;
Signe gateway, dashboard remains inactive).

Rollback: restore `config.yaml.before` from the target backup, `git revert` the
backport commit (does not touch Mesh files), restart the same active services.
The target backup includes previous HEAD and protected-file digests. OAuth state,
identity, session stores, and cron records are not modified by deployment.
