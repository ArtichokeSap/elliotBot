---
name: session-context
description: "Maintains continuity across AI agent sessions via structured logging, tiered context loading, and automatic archival. Triggered when: (1) user says 'log session', 'save session', 'end session', 'record session', (2) user says 'resume session', 'continue session', 'catch me up', 'where were we', (3) user says 'quick context' for summary only. Proactively suggests session logging after substantial work."
---

# Session Context Skill

Maintains continuity across AI agent sessions by logging work, archiving history, and enabling fast context resumption. Solves the problem of losing context between chat sessions.

## File Locations

```
.agents/skills/session-context/
├── SKILL.md             # This file - skill definition and behavior
├── config.yaml          # Skill configuration (archival, triggers, etc.)
├── summary.md           # Rolling summary (ALWAYS read first - ~500 tokens)
├── recent/              # Last 7 days of sessions
├── archive/             # Monthly compressed archives
└── _templates/          # Session and summary templates
```

## Mode Selection

Determine which mode applies:

**Logging a session?** User wants to save current work, end session, or record progress.
- Follow: SESSION LOGGING PROCEDURE below

**Resuming a session?** User wants to continue previous work, load context, or catch up.
- Follow: SESSION RESUMPTION PROCEDURE below

**Quick context?** User just wants a fast summary without git checks.
- Read only `summary.md`, report current state to user.

**Proactive suggestion?** After substantial work (5+ file edits, major decisions, complex debugging), suggest:
> "We've done significant work this session. Say 'log session' to preserve context for future sessions."

## Session Logging Procedure

When triggered by: "log session", "save session", "end session", "record session"

1. **Create session file**: `.agents/skills/session-context/recent/YYYY-MM-DD-[brief-title].md`
2. **Use template**: `.agents/skills/session-context/_templates/session.md`
3. **Fill in**: Summary, Goals, Changes, Decisions, Problems, Next Steps, AI Context
4. **Update summary.md**: Add entry to recent sessions table, update current state
5. **Check archival**: If any sessions in `recent/` are >7 days old, archive them
6. **Confirm with user**: Show file path, offer adjustments

**What NOT to Log**: Trivial Q&A, sessions user declines to log

## Session Resumption Procedure

When triggered by: "resume session", "continue session", "catch me up", "where were we"

**Tiered Context Loading** (optimizes context window):

| Tier | Files | Tokens | When |
|------|-------|--------|------|
| 1 | `summary.md` | ~500 | Always |
| 2 | `recent/*.md` | ~1500 | Full resume |
| 3 | `archive/*.md` | Variable | Historical search |

**For "quick context"**: Read only `summary.md`, report to user.

**For full resume**:
1. Read `summary.md` (always)
2. Run `git status` and `git diff --stat`
3. Read recent sessions if needed for detail
4. Summarize: state, next steps, blockers
5. Offer options from next steps

## Archival (Prevents Unbounded Growth)

**Automatic rules** (configured in `config.yaml`):
- Sessions >7 days old: Move from `recent/` to `archive/`
- Archive format: Monthly summaries (`archive/YYYY-MM.md`)
- Max 12 monthly archives, then compress to yearly

**When archiving**:
1. Extract: date, title, key accomplishments, decisions
2. Append to monthly archive file
3. Delete original from `recent/`
4. Regenerate `summary.md` if needed

## Portability

This skill is designed to work in **any repository**. To install in a new project:

1. Copy the entire `.agents/skills/session-context/` folder to your repo
2. Clear example sessions: `rm -rf .agents/skills/session-context/recent/* .agents/skills/session-context/archive/*`
3. Customize `summary.md` for your project
4. Commit: `git add .agents/skills/session-context && git commit -m "feat: Add session context skill"`

No changes to `.github/copilot-instructions.md` are required — agents that support the `.agents/skills/` convention will discover this skill automatically.

## Design References

- **Architecture**: See `docs/reference/SESSION_CONTEXT_SKILL_DESIGN.md` for full design rationale
- **Export Guide**: See `docs/reference/SESSION_CONTEXT_EXPORT_GUIDE.md` for cross-project usage
