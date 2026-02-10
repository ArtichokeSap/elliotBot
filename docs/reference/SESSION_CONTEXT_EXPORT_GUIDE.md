# Session Context Skill - Export/Installation Guide

## What This Is

The Session Context Skill is a **behavioral pattern** for AI agents that maintains continuity across chat sessions. It follows the `.agents/skills/` standard format and consists of:

1. **Skill definition** (`.agents/skills/session-context/SKILL.md`) - Standard-format skill with YAML frontmatter
2. **Data files** (`.agents/skills/session-context/`) - Templates, config, session logs

## Current State of AI Skills

> ℹ️ **As of 2026**, the `.agents/skills/` directory convention has been adopted by thousands of repositories (lobehub, nitrojs, wagmi, and many others). Skills are distributed as folders you copy into your repository's `.agents/skills/` directory.

| What Exists | What's Emerging |
|-------------|-----------------|
| `.agents/skills/` directory standard | Skill registries/marketplaces |
| `SKILL.md` with YAML frontmatter | Version management |
| Copy-paste distribution | Cross-repo skill updates |
| Per-repo configuration | Automated skill installers |

## How to Export (Use in Another Project)

### Step 1: Copy the Skill Folder

From this repository, copy the entire `.agents/skills/session-context/` folder:

```bash
# In your target repository
mkdir -p .agents/skills

# Copy from Elliott Wave Bot (or download from GitHub)
cp -r /path/to/elliotBot/.agents/skills/session-context .agents/skills/session-context
```

**Files you'll get:**
```
.agents/skills/session-context/
├── SKILL.md             # Skill definition (standard format)
├── config.yaml          # Configuration (edit for your needs)
├── summary.md           # Rolling summary (will be project-specific)
├── recent/              # Recent session logs
├── archive/             # Archived sessions
└── _templates/
    ├── session.md       # Template for new sessions
    └── summary.md       # Template for summary file
```

### Step 2: Initialize for Your Project

1. **Clear the example sessions**:
   ```bash
   rm -rf .agents/skills/session-context/recent/*
   rm -rf .agents/skills/session-context/archive/*
   ```

2. **Customize `summary.md`** for your project:
   ```bash
   # Edit .agents/skills/session-context/summary.md
   # Replace Elliott Wave Bot content with your project's context
   ```

3. **Commit to your repo**:
   ```bash
   git add .agents/skills/session-context
   git commit -m "feat: Add session context skill for AI continuity"
   ```

### Step 3: Verify It Works

Start a new Copilot session and say:
- "quick context" - Should read and summarize your summary.md
- "log session" - Should create a new session file

> **Note**: No changes to `.github/copilot-instructions.md` are required. Agents that
> support the `.agents/skills/` convention will discover `SKILL.md` automatically.

## Keeping Skills Updated

Since there's no package manager yet, updates are manual:

1. Check the source repo for changes to `.agents/skills/session-context/`
2. Copy updated `SKILL.md` and `config.yaml`
3. Your session data (`summary.md`, `recent/`, `archive/`) remains untouched

---

## FAQ

**Q: Why `.agents/skills/` instead of `.ai/session-context/`?**

A: The `.agents/skills/` directory is an emerging standard used by thousands of repositories. Each skill gets its own directory with a `SKILL.md` entry point. This makes skills discoverable, portable, and compatible with AI agent platforms.

**Q: Can I have multiple skills in one repo?**

A: Yes! Each skill is its own folder under `.agents/skills/`:
```
.agents/skills/
├── session-context/     # This skill
├── code-review/         # Another skill
└── testing/             # Yet another
```

**Q: What if I want to modify the skill for my project?**

A: Go ahead! Each installation is independent. Customize `config.yaml`, templates, or even `SKILL.md`.

**Q: Will changes to Elliott Wave Bot's skill affect my copy?**

A: No. Once you copy the files, they're independent. You'd need to manually sync updates.
