# Copilot Instructions for Elliott Wave Bot

## Session Context Integration

This project includes an advanced session context management system that provides persistent, portable context across development sessions.

### Trigger Phrases
When users type any of these phrases, activate the session context skill:
- "resume session"
- "continue session"
- "session resume"
- "pick up where we left off"

### Context Loading (Tiered System)

#### Tier 1: Always Load (Summary)
Read `.ai/session-context/summary.md` first - this contains:
- Current project state
- Active decisions
- Next steps prioritized
- Quick reference commands

#### Tier 2: On Resume Request (Recent Sessions)
Load recent session files from `.ai/session-context/recent/`:
- Detailed session logs
- Code changes made
- Decision rationales
- Implementation progress

#### Tier 3: Historical Context (On Demand)
Access archived sessions from `.ai/session-context/archive/` for:
- Long-term project evolution
- Pattern recognition
- Historical decisions

### Response Guidelines

#### For Resume Requests
1. Acknowledge the resume request
2. Summarize current state from summary.md
3. Highlight next steps and blockers
4. Offer specific continuation options
5. Reference recent work done

#### For General Development
- Maintain context of ongoing work
- Reference active decisions from summary
- Suggest next logical steps
- Avoid repeating completed work

### File Structure Awareness
```
.ai/session-context/
├── skill.yaml           # Configuration
├── summary.md           # Rolling summary (ALWAYS read)
├── recent/              # Last 7 days
├── archive/             # Monthly archives
└── _templates/          # Reusable templates

docs/
├── reference/           # Design docs and guides
└── sessions/            # Session logs
```

### Integration Commands
- Use `git status` and `git diff --stat` to check current changes
- Run tests with `python -m pytest` when appropriate
- Health check: `python validate_imports.py`
- Full analysis: `python main.py analyze AAPL`

### Error Handling
- If context files missing: Gracefully continue with available information
- If imports fail: Suggest running `pip install -r requirements.txt`
- If tests fail: Focus on data_loader implementation first

### Development Workflow
1. Load session context on resume
2. Check git status for uncommitted changes
3. Review next steps from summary.md
4. Implement highest priority items
5. Update summary.md with progress
6. Commit changes with descriptive messages

### Key Project Areas
- **Data Loading**: Yahoo Finance + Binance support (currently missing)
- **Wave Detection**: Elliott Wave pattern recognition
- **Visualization**: Interactive charts with Plotly
- **Backtesting**: Strategy validation engine
- **ML Integration**: Pattern recognition models

### Current Blockers
- `src/data/data_loader.py` needs implementation
- Test suite expects DataLoader class
- Main.py imports failing due to missing module

Prioritize implementing the data loading functionality to unblock the entire pipeline.