## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Per-turn issue declaration (mooladhara#131) — NON-NEGOTIABLE

Every assistant turn MUST begin with a leading tag as the first token of the first line:

| Tag | Meaning |
|---|---|
| `[#123]` | Work bound to issue 123. **The normal case.** |
| `[unbound]` | **WORK with no issue yet — a TRIPWIRE, not a resting state.** File an issue. |
| `[status]` | Not work: a status/progress/report answer. Legitimate, no issue needed. |
| `[chat]` | Not work: conversation, clarification, a question answered. Legitimate. |
| `[meta]` | Not work: a quick command/lookup with no work product. Legitimate. |

ONE primary issue per turn. A turn spanning two issues is a signal to **split the turn**.

**Why:** it converts issue↔work attribution from *inference* to *declaration*. Unbound work
is unbillable and untraceable. `[status]`/`[chat]`/`[meta]` exist so that non-work turns do
not have to lie (`[#N]` on a status query) or trip a false alarm — a convention people must
game is a convention they will ignore.

A Stop hook (`.claude/hooks/check_issue_tag.py`) enforces the tag's presence. It is
**advisory** — it warns, it never blocks.
