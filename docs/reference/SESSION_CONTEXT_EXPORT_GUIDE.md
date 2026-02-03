# Session Context Skill - Export/Installation Guide

## What This Is

The Session Context Skill is a **behavioral pattern** for AI agents that maintains continuity across chat sessions. It consists of:

1. **Data files** (`.ai/session-context/`) - Templates, config, session logs
2. **Instructions** (`.github/copilot-instructions.md`) - Behavioral rules the AI follows

## Current State of AI Skills

> ⚠️ **Important**: As of 2026, there is no standard package manager for AI agent skills. Skills are distributed as files you copy into your repository.

| What Exists | What Doesn't Exist (Yet) |
|-------------|-------------------------|
| File-based skills | `copilot skill install session-context` |
| Manual installation | Skill registries/marketplaces |
| Copy-paste distribution | Version management |
| Per-repo configuration | Cross-repo skill updates |

## How to Export (Use in Another Project)

### Step 1: Copy the Skill Files

From this repository, copy the entire `.ai/session-context/` folder:

```bash
# In your target repository
mkdir -p .ai/session-context

# Copy from Elliott Wave Bot (or download from GitHub)
cp -r /path/to/elliotBot/.ai/session-context/* .ai/session-context/
```

**Files you'll get:**
```
.ai/session-context/
├── skill.yaml           # Configuration (edit for your needs)
├── summary.md           # Rolling summary (will be project-specific)
├── recent/              # Recent session logs
├── archive/             # Archived sessions
└── _templates/
    ├── session.md       # Template for new sessions
    └── summary.md       # Template for summary file
```

### Step 2: Add Instructions to Your Copilot Config

Copy the skill instructions into your project's `.github/copilot-instructions.md`.

If you don't have this file yet:
```bash
mkdir -p .github
touch .github/copilot-instructions.md
```

Then add this section (copy from Elliott Wave Bot's file, the section starting with `## 🔴 SESSION CONTEXT SKILL`):

```markdown
## 🔴 SESSION CONTEXT SKILL (Generalized)

> **Skill Design**: See `docs/reference/SESSION_CONTEXT_SKILL_DESIGN.md` for full architecture.
> **Configuration**: See `.ai/session-context/skill.yaml` for settings.

### File Locations
... (copy the full section)
```

### Step 3: Initialize for Your Project

1. **Clear the example sessions**:
   ```bash
   rm -rf .ai/session-context/recent/*
   rm -rf .ai/session-context/archive/*
   ```

2. **Customize `summary.md`** for your project:
   ```bash
   # Edit .ai/session-context/summary.md
   # Replace Elliott Wave Bot content with your project's context
   ```

3. **Commit to your repo**:
   ```bash
   git add .ai/session-context .github/copilot-instructions.md
   git commit -m "feat: Add session context skill for AI continuity"
   ```

### Step 4: Verify It Works

Start a new Copilot session and say:
- "quick context" - Should read and summarize your summary.md
- "log session" - Should create a new session file

## Keeping Skills Updated

Since there's no package manager, updates are manual:

1. Check the source repo for changes
2. Copy updated files
3. Merge any instruction changes

## Future Vision

When/if GitHub adds skill support, this might become:
```yaml
# .github/copilot-skills.yaml (hypothetical)
skills:
  - name: session-context
    version: "1.0.0"
    source: github.com/some-org/copilot-skills
```

For now, file copying is the way.

---

## FAQ

**Q: Is Elliott Wave Bot "using" or "containing" the skill?**

A: Both! The skill is embedded in the repo. When Copilot opens Elliott Wave Bot, it automatically has the skill available because the instructions are in `.github/copilot-instructions.md`.

**Q: Can I have multiple skills in one repo?**

A: Yes! Add more sections to `copilot-instructions.md` and more folders under `.ai/`.

**Q: What if I want to modify the skill for my project?**

A: Go ahead! Each installation is independent. Customize `skill.yaml`, templates, or even the instructions.

**Q: Will changes to Elliott Wave Bot's skill affect my copy?**

A: No. Once you copy the files, they're independent. You'd need to manually sync updates.
