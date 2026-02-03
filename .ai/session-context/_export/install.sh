#!/bin/bash
# Session Context Skill - Quick Install Script
#
# Usage: curl -sL <url> | bash
# Or: ./install.sh /path/to/target/repo
#
# This creates the skill structure and reminds you to add instructions.

set -e

TARGET="${1:-.}"

echo "📦 Installing Session Context Skill..."
echo "   Target: $TARGET"

# Create directory structure
mkdir -p "$TARGET/.ai/session-context/recent"
mkdir -p "$TARGET/.ai/session-context/archive"
mkdir -p "$TARGET/.ai/session-context/_templates"

# Create skill.yaml
cat > "$TARGET/.ai/session-context/skill.yaml" << 'EOF'
# Session Context Skill Configuration
version: "1.0"

paths:
  base: ".ai/session-context"
  summary: "summary.md"
  recent: "recent"
  archive: "archive"
  templates: "_templates"

archival:
  recent_days: 7
  archive_after_days: 30
  archive_strategy: "monthly"
  max_archive_files: 12

summary:
  max_recent_entries: 10
  max_tokens_estimate: 1000

triggers:
  log_session: ["log session", "save session", "end session"]
  resume_session: ["resume session", "continue session", "catch me up"]
  quick_context: ["quick context"]
EOF

# Create session template
cat > "$TARGET/.ai/session-context/_templates/session.md" << 'EOF'
# Session: {{DATE}} - {{TITLE}}

## Summary
<!-- One paragraph: what was accomplished -->

## Goals
- [ ] Goal 1
- [x] Completed goal (example)

## Changes Made
- `path/file.py` - Description

## Decisions
- **Decision**: X over Y
- **Reason**: Because...

## Problems → Solutions
- Problem: ...
  - Solution: ...

## Next Steps
1. Priority task
2. Lower priority

## AI Context
<!-- For the next AI session -->
- Current state: ...
- Blocked on: ...
EOF

# Create summary template
cat > "$TARGET/.ai/session-context/_templates/summary.md" << 'EOF'
# Project Context Summary
*Auto-updated from session logs. Last updated: {{DATE}}*

## Current State
- **Main branch**: Status
- **In-progress**: Current work
- **Blocked**: Any blockers

## Recent Sessions
| Date | Title | Key Changes |
|------|-------|-------------|
| YYYY-MM-DD | Title | Brief summary |

## Active Decisions
- Decision 1 (decided YYYY-MM-DD)

## Next Steps (Prioritized)
1. [ ] Highest priority
2. [ ] Medium priority

## Quick Reference
- Test command: `...`
- Build command: `...`
EOF

# Create initial summary.md
cat > "$TARGET/.ai/session-context/summary.md" << 'EOF'
# Project Context Summary
*Auto-updated from session logs. Last updated: (not yet)*

## Current State
- **Main branch**: Unknown (customize this)
- **In-progress**: None yet
- **Blocked**: None

## Recent Sessions
| Date | Title | Key Changes |
|------|-------|-------------|
| (none yet) | | |

## Active Decisions
- (none documented yet)

## Next Steps (Prioritized)
1. [ ] Customize this summary for your project
2. [ ] Add copilot instructions (see INSTRUCTIONS.md)

## Quick Reference
- Test command: `(customize)`
- Build command: `(customize)`
EOF

echo ""
echo "✅ Skill files created!"
echo ""
echo "⚠️  IMPORTANT: You must also add the behavioral instructions."
echo ""
echo "   1. Open/create: $TARGET/.github/copilot-instructions.md"
echo "   2. Copy the content from: .ai/session-context/_export/INSTRUCTIONS.md"
echo "   3. Customize summary.md for your project"
echo ""
echo "   Then commit everything:"
echo "   git add .ai/session-context .github/copilot-instructions.md"
echo "   git commit -m 'feat: Add session context skill'"
echo ""
