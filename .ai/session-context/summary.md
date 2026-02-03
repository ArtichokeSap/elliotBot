# Elliott Wave Bot - Project Context Summary
*Auto-updated from session logs. Last updated: 2026-02-03*

## Current State
- **Main branch**: Stable with security fixes, test suite in place
- **In-progress**: Generalizing session context skill
- **Blocked**: None

## Recent Sessions
| Date | Title | Key Changes |
|------|-------|-------------|
| 2026-02-03 | Session Context Generalization | Designed portable session skill with archival |
| 2026-01-31 | Session Logging System | Created docs structure, added trigger phrases |
| 2026-01-29 | Security & Tests | Fixed pickle vulnerability, added 40 tests |

## Active Decisions
- Use `joblib.load()` not `pickle.load()` for security (2026-01-29)
- Whitelist approach for input validation (2026-01-29)
- Three-folder doc structure: reference/changelog/sessions (2026-01-31)
- Session context lives in `.ai/session-context/` (2026-02-03)
- Rolling summary.md for constant-size context (2026-02-03)

## Next Steps (Prioritized)
1. [ ] Get remaining tests passing (API adjustments needed)
2. [ ] Add test coverage reporting
3. [ ] Continue Elliott Wave bot feature development
4. [ ] Consider pre-commit hooks

## Quick Reference
- Test command: `pytest` or `python -m pytest -v`
- Health check: `python tools/quick_test.py`
- Full analysis: `python main.py`
- Key config: `config.yaml` (copy from `config_template.yaml`)
