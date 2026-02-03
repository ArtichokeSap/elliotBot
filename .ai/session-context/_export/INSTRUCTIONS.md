# Session Context Skill - Copilot Instructions
# 
# INSTALLATION: Merge this content into your .github/copilot-instructions.md
# 
# This file contains the behavioral instructions that teach the AI how to
# log and resume sessions. Without these, the .ai/session-context/ folder
# is just data with no behavior.

---

## 🔴 SESSION CONTEXT SKILL (Generalized)

> **Configuration**: See `.ai/session-context/skill.yaml` for settings.

### File Locations
```
.ai/session-context/
├── skill.yaml           # Skill configuration
├── summary.md           # Rolling summary (ALWAYS read first - ~500 tokens)
├── recent/              # Last 7 days of sessions
├── archive/             # Monthly compressed archives
└── _templates/          # Session and summary templates
```

### Trigger Phrases
**Logging**: "log session", "save session", "end session", "record session"
**Resumption**: "resume session", "continue session", "catch me up", "where were we"
**Quick**: "quick context" (summary only, no git checks)

---

### 🔴 SESSION LOGGING PROCEDURE

When triggered:

1. **Create session file**: `.ai/session-context/recent/YYYY-MM-DD-[brief-title].md`
2. **Use template**: `.ai/session-context/_templates/session.md`
3. **Fill in**: Summary, Goals, Changes, Decisions, Problems, Next Steps, AI Context
4. **Update summary.md**: Add entry to recent sessions table, update current state
5. **Check archival**: If any sessions in `recent/` are >7 days old, archive them
6. **Confirm with user**: Show file path, offer adjustments

**What NOT to Log**: Trivial Q&A, sessions user declines

---

### 🟢 SESSION RESUMPTION PROCEDURE

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

---

### 📦 ARCHIVAL (Prevents Unbounded Growth)

**Automatic rules** (configured in `skill.yaml`):
- Sessions >7 days old: Move from `recent/` to `archive/`
- Archive format: Monthly summaries (`archive/YYYY-MM.md`)
- Max 12 monthly archives, then compress to yearly

**When archiving**:
1. Extract: date, title, key accomplishments, decisions
2. Append to monthly archive file
3. Delete original from `recent/`
4. Regenerate `summary.md` if needed
