# Session Context Skill - Generalized Design

## Executive Summary

This document proposes a generalized, reusable "Session Context" skill for AI agents that maintains continuity across sessions. It addresses the lessons learned from the Elliott Wave Bot implementation and re-architects the solution to be:

1. **Portable** - Works in any repository, not just this one
2. **Scalable** - Handles folder growth with rotation/archival
3. **Context-Efficient** - Balances getting "up to speed" without consuming the entire context window

---

## Is This a "Skill"? Terminology Clarification

### What We Mean by "Skill"

In the AI agent ecosystem, there are several concepts:

| Term | Definition | Example |
|------|------------|---------|
| **Tool** | A specific function the AI can invoke | `bash`, `edit`, `web_fetch` |
| **Capability** | A broad ability enabled by tools | "Can read files", "Can execute code" |
| **Skill** | A **reusable behavioral pattern** that combines tools + instructions + domain knowledge | "Session Context Management" |
| **Agent** | An AI instance with tools, capabilities, and skills configured | GitHub Copilot Agent |

**Session Context is a Skill** because it:
- Combines multiple tools (file read/write, git commands)
- Requires specific behavioral instructions (when to log, what format)
- Has domain knowledge (where files go, how to summarize)
- Is portable across different projects/contexts

### How Skills Get "Installed"

Currently, skills are installed via:
1. **Instruction files** (`.github/copilot-instructions.md`) - Behavioral rules
2. **File templates** - Structure for outputs
3. **Convention** - File locations, naming patterns

Future platforms may support:
- Skill packages with metadata
- Declarative skill definitions
- Skill marketplaces/registries

---

## Current Implementation: Problems Identified

### Problem 1: Unbounded Growth
```
docs/sessions/
├── 2026-01-29-security-fixes.md      # 2KB
├── 2026-01-31-session-logging.md     # 2KB
├── 2026-02-01-feature-x.md           # 2KB
├── ...
└── 2027-01-01-session-365.md         # 730KB total!
```

After a year of daily development, you'd have 365+ files consuming storage and making directory listings unwieldy.

### Problem 2: Context Window Consumption

When resuming:
```
# Current approach: Read "most recent 1-2 session files"
Session 1: ~2000 tokens
Session 2: ~2000 tokens
─────────────────────
Total: ~4000 tokens just for context!
```

With an 8K-32K context window, 4K tokens for "catching up" leaves less room for actual work.

### Problem 3: Non-Portable

Current implementation:
- Paths are hardcoded (`docs/sessions/`)
- Template is project-specific
- Instructions embedded in project-specific file

### Problem 4: No Relevance Filtering

Old sessions about "Fixed typo in README" get equal weight to "Redesigned entire authentication system" when the AI is trying to understand context.

---

## Re-Architected Design: "Session Context Skill v2"

### Core Principles

1. **Tiered Summaries** - Not all context is equal; use compression
2. **Automatic Archival** - Old sessions compress, rotate, or archive
3. **Relevance Scoring** - More recent + more impactful = higher priority
4. **Portable Configuration** - Skill config separate from project config
5. **Lazy Loading** - Don't read everything; read what's needed

### Proposed File Structure

> **Note**: As of v1.1, this skill has been migrated to the `.agents/skills/` standard
> format (see below). The original `.ai/session-context/` layout is preserved here for
> historical context.

```
.agents/skills/session-context/         # Standard .agents/skills location
├── SKILL.md                            # Skill definition (standard format)
├── config.yaml                         # Skill configuration
├── summary.md                          # Rolling summary (always read)
├── recent/                             # Recent sessions (last 7 days)
│   ├── 2026-02-01-feature-x.md
│   └── 2026-02-03-bug-fix.md
├── archive/                            # Compressed older sessions
│   ├── 2026-01.md                      # January's sessions, summarized
│   └── 2026-Q1.md                      # Q1 aggregate (optional)
└── _templates/                         # Session and summary templates
    ├── session.md
    └── summary.md
```

### Key Innovation: The Rolling Summary

**`summary.md`** is a single file that:
- Always gets read on session resume (it's small)
- Contains the "executive summary" of project state
- Auto-updates when sessions are logged
- Replaces reading multiple full session files

```markdown
# Project Context Summary
*Auto-generated from session logs. Last updated: 2026-02-03*

## Current State
- Main branch: stable, all tests passing
- In-progress: Adding WebSocket support (branch: feature/websocket)
- Blocked: Waiting for API key for external service

## Recent Sessions (7 days)
| Date | Title | Key Changes |
|------|-------|-------------|
| 2026-02-03 | Bug Fix | Fixed auth token expiry |
| 2026-02-01 | Feature X | Added user preferences API |

## Active Decisions
- Using PostgreSQL over MongoDB (decided 2026-01-15)
- REST API, not GraphQL (decided 2026-01-10)

## Key Contacts/Context
- Security review required before v2.0 release
- Performance target: <100ms API response time
```

**Size**: ~500-1000 tokens (vs 4000+ for reading multiple sessions)

### Tiered Reading Strategy

```
┌─────────────────────────────────────────────────────────────┐
│ TIER 1: Always Read (~500 tokens)                           │
│ • summary.md - Rolling project state                        │
│ • current.md - If exists, in-progress session               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (if user wants more detail)
┌─────────────────────────────────────────────────────────────┐
│ TIER 2: On-Demand (~1500 tokens)                            │
│ • recent/*.md - Last 7 days of sessions                     │
│ • Only read if user asks "catch me up fully"                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (if searching for something specific)
┌─────────────────────────────────────────────────────────────┐
│ TIER 3: Archived (~deep dive)                               │
│ • archive/*.md - Monthly/quarterly summaries                │
│ • Only read if searching for historical decision           │
└─────────────────────────────────────────────────────────────┘
```

### Automatic Archival Rules

```yaml
# .agents/skills/session-context/config.yaml
archival:
  recent_days: 7           # Sessions younger than this stay in recent/
  archive_after_days: 30   # Sessions older than this get archived
  archive_strategy: monthly # Combine into monthly summaries
  max_archive_files: 12    # Keep only 12 months, then consolidate to yearly

summary:
  max_recent_entries: 10   # Only show 10 sessions in summary table
  include_decisions: true  # Track key decisions
  include_blockers: true   # Track current blockers
```

### Session Archival Process

When a session is archived:

1. **Extract key info**: Title, date, main accomplishments, decisions
2. **Add to monthly summary**: `archive/2026-01.md`
3. **Remove from recent/**: Delete the full file
4. **Update summary.md**: Refresh the rolling summary

```python
# Pseudocode for archival
def archive_session(session_file):
    content = read(session_file)
    key_info = extract_summary(content)  # AI-generated or template-based
    
    # Extract date from filename (e.g., "2026-01-29-security.md" -> "2026-01")
    filename = os.path.basename(session_file)
    date_str = filename[:7]  # "YYYY-MM"
    month_file = f"archive/{date_str}.md"
    
    append_to_archive(month_file, key_info)
    delete(session_file)
    regenerate_summary()
```

---

## Portable Skill Definition

### Skill Format (Current Standard)

The `.agents/skills/` format is now the standard for AI agent skills. Each skill lives in
its own directory under `.agents/skills/<skill-name>/` with a `SKILL.md` file as the
entry point:

```yaml
# .agents/skills/session-context/SKILL.md (YAML frontmatter)
---
name: session-context
description: "Maintains continuity across AI agent sessions..."
---
```

This format is used by thousands of repositories (lobehub, nitrojs, wagmi, and many others)
and is supported by GitHub Copilot and other AI agent platforms.

### Installation as Standalone Skill

To use in any repository:

```bash
# Copy the skill folder to your repo
cp -r .agents/skills/session-context /path/to/your/repo/.agents/skills/session-context

# Clear example sessions
rm -rf .agents/skills/session-context/recent/*
rm -rf .agents/skills/session-context/archive/*

# Customize summary.md for your project, then commit
git add .agents/skills/session-context
git commit -m "feat: Add session context skill"
```

---

## Context Window Optimization Details

### Token Budget Example

For a 32K token context window:

| Component | Token Budget | Purpose |
|-----------|--------------|---------|
| System prompt | ~2000 | Core AI instructions |
| **Session context** | **~1000** | Project state (our skill) |
| Tools/Functions | ~1000 | Available tool definitions |
| User messages | ~4000 | Current conversation |
| Working memory | ~24000 | Files, code, analysis |

**Goal**: Keep session context under 1000 tokens while maximizing usefulness.

### Summary Compression Techniques

1. **Bullet points over prose**: "Added user auth" not "In this session we worked on adding user authentication"

2. **Skip boilerplate**: Don't include empty template sections

3. **Use tables for lists**: More compact than markdown lists

4. **Abbreviate paths**: `src/auth/` not `/home/user/project/src/auth/`

5. **Reference, don't repeat**: "See 2026-01 archive for auth decisions" not full history

### Dynamic Context Loading

```python
# Pseudocode for smart loading
def load_session_context(request_type):
    context = []
    
    # Always load summary (small)
    context.append(read("summary.md"))  # ~500 tokens
    
    if request_type == "quick_context":
        return context  # Done!
    
    if request_type == "full_resume":
        # Load recent sessions
        for session in get_recent_sessions(days=7):
            context.append(read(session))  # ~1500 tokens
    
    if request_type == "search_history":
        # User is looking for something specific
        query = get_user_query()
        relevant = search_archives(query)
        context.append(relevant)  # Variable tokens
    
    return context
```

---

## Implementation Roadmap

### Phase 1: Minimal Viable Skill ✅
- [x] Design document (this file)
- [x] Create folder structure with templates
- [x] Migrate existing sessions
- [x] Create rolling `summary.md`
- [x] Update instructions for new paths

### Phase 1.5: Migrate to `.agents/skills/` Standard ✅
- [x] Move from `.ai/session-context/` to `.agents/skills/session-context/`
- [x] Create `SKILL.md` with standard YAML frontmatter format
- [x] Remove embedded instructions from `copilot-instructions.md`
- [x] Update all documentation references

### Phase 2: Automatic Archival
- [ ] Add archival logic (could be a daily cron or on-log trigger)
- [ ] Create monthly archive format
- [ ] Implement summary regeneration

### Phase 3: Cross-Project Portability
- [x] Extract skill as standalone copyable folder
- [x] Create installation instructions
- [ ] Test in different repository types

### Phase 4: Platform Integration (Future)
- [ ] Propose as official Copilot skill
- [x] Adopt `.agents/skills/` standard format
- [ ] Build skill discovery/registry

---

## Comparison: Before vs After

| Aspect | Before (v1) | After (v1.1 - Current) |
|--------|-------------|------------------------|
| Location | `.ai/session-context/` | `.agents/skills/session-context/` |
| Skill format | Custom `skill.yaml` | Standard `SKILL.md` with frontmatter |
| Folder growth | Unbounded | Archived monthly |
| Context tokens | ~4000 | ~500-1000 |
| Portability | Project-specific | Copyable folder, standard format |
| Resume speed | Read 2 full files | Read 1 summary |
| Historical access | All equal weight | Tiered priority |
| Configuration | Hardcoded | YAML config |
| Discovery | Manual instructions in copilot-instructions.md | Auto-discovered by `.agents/skills/` convention |

---

## Appendix: Alternative Approaches Considered

### A. Database Instead of Files
**Pros**: Efficient queries, structured data
**Cons**: Adds dependency, not git-tracked, not human-readable
**Decision**: Stick with files for simplicity and git integration

### B. Single Monolithic Log
**Pros**: One file to read
**Cons**: Gets huge, can't easily search/filter
**Decision**: Tiered structure is better

### C. AI-Generated Summaries Only
**Pros**: Maximum compression
**Cons**: Loses detail, AI hallucination risk
**Decision**: Keep structured logs, AI generates summary.md

### D. Store in Git Commit Messages
**Pros**: Built into workflow, no extra files
**Cons**: Not enough space, hard to read, not searchable
**Decision**: Separate files are better

---

## Conclusion

The Session Context Skill can absolutely be generalized and made portable. The key innovations are:

1. **Rolling summary.md** - Constant-size context that always gets read
2. **Automatic archival** - Prevents unbounded growth
3. **Tiered loading** - Balances detail vs context window
4. **Portable config** - Works in any repository

This pattern could become a standard skill for AI agents, potentially distributed as a package or built into agent platforms.
