# Session Context Export Guide

## Overview
This guide explains how to export and import session context for the Elliott Wave Bot project.

## Export Process

### Automatic Export
The session context skill automatically exports context when:
- Session ends gracefully
- Major changes are committed
- Manual export is triggered

### Manual Export
```bash
# Export current session context
python tools/export_session.py

# Export with specific date range
python tools/export_session.py --start 2026-01-01 --end 2026-02-02
```

## Import Process

### Automatic Import
Session context is automatically loaded when:
- "resume session" is typed
- VS Code workspace is opened
- Copilot requests context

### Manual Import
```bash
# Import session context
python tools/import_session.py --file session_context_2026-02-02.json

# Import from URL
python tools/import_session.py --url https://example.com/session.json
```

## File Structure

```
.ai/session-context/
├── skill.yaml           # Skill configuration
├── summary.md           # Rolling summary (always read first)
├── recent/              # Last 7 days sessions
├── archive/             # Monthly compressed archives
└── _templates/          # Session and summary templates
```

## Configuration

Edit `.ai/session-context/skill.yaml` to customize:

```yaml
# Core settings
enabled: true
auto_archive: true
max_recent_sessions: 7

# File locations
summary_file: ".ai/session-context/summary.md"
recent_dir: ".ai/session-context/recent/"
archive_dir: ".ai/session-context/archive/"
```

## Troubleshooting

### Context Not Loading
1. Check `.ai/session-context/skill.yaml` exists
2. Verify `enabled: true`
3. Check file permissions
4. Review VS Code Copilot logs

### Export Failures
1. Check disk space
2. Verify write permissions
3. Check for file locks
4. Review error logs

### Import Errors
1. Validate JSON format
2. Check file paths exist
3. Verify version compatibility
4. Check for conflicts

## Best Practices

- Export regularly during development
- Keep summary.md under version control
- Archive old sessions monthly
- Test imports after major changes
- Document custom configurations
