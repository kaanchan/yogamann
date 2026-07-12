#!/usr/bin/env python3
"""Stop hook — enforce the per-turn issue-declaration convention (mooladhara #131).

Ported from kaanchan/time-track-reporter (its #83), where it is proven in production.
This version EXTENDS the tag vocabulary so the convention stays honest instead of being
gamed: the original had only [#N] and [unbound], which forced status queries and one-line
commands to either lie ([#N] on non-work) or trip a "file an issue NOW" alarm that does
not apply. Both failure modes teach people to ignore the tag.

THE TAG — first token of the first line of every assistant turn:

    [#123]      Work bound to issue 123. The normal case.
    [unbound]   WORK with no issue yet. A TRIPWIRE, not a resting state -- file an issue.
    [status]    Not work: a status/progress/report answer. Legitimate. No issue needed.
    [chat]      Not work: conversation, clarification, a question answered. Legitimate.
    [meta]      Not work: a quick command/lookup with no work product (ls, git status, ...).

Why the split matters for BILLING: [unbound] means "billable work happened with nothing to
bill it to" -- that is the leak. [status]/[chat]/[meta] mean "no billable work product" --
that is not a leak, and must not be flagged as one, or the signal drowns in noise.

ONE primary issue per turn. A turn spanning two issues is a signal to SPLIT THE TURN.

ADVISORY, never blocking: always exits 0. A missing tag prints a fail-loud reminder to
stderr. FAIL-OPEN: any internal problem (no stdin, unreadable transcript, parse error)
prints one 'skipped (reason)' line and exits 0. An advisory check must never wedge a session.

PRIVACY: reads only enough of the last assistant message to test the leading token. It never
inspects, stores, or emits message content beyond a 40-char preview in the warning.

Wire into .claude/settings.json under hooks.Stop:
    {"type": "command", "command": "python .claude/hooks/check_issue_tag.py"}
"""
import json
import re
import sys

# [#123] -> issue-bound work
ISSUE_TAG = re.compile(r"^\s*\[#(\d+)\]")
# [unbound] -> WORK without an issue. Tripwire.
UNBOUND_TAG = re.compile(r"^\s*\[unbound\]", re.IGNORECASE)
# [status] / [chat] / [meta] -> legitimately not work. Not a tripwire.
NONWORK_TAG = re.compile(r"^\s*\[(status|chat|meta)\]", re.IGNORECASE)


def leading_text(record):
    """First text block of an assistant record, or None. Reads only enough to test the tag."""
    content = (record.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text") or ""
    return None


def last_assistant_text(transcript_path):
    """Leading text of the LAST assistant turn in the transcript .jsonl, or None."""
    last = None
    with open(transcript_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "assistant":
                continue
            text = leading_text(rec)
            if text is not None:
                last = text
    return last


def main():
    raw = sys.stdin.read()
    if not raw.strip():
        print("check_issue_tag: skipped (no hook input on stdin)", file=sys.stderr)
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"check_issue_tag: skipped (unparseable hook input: {e})", file=sys.stderr)
        return 0

    if payload.get("stop_hook_active"):
        return 0

    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        print("check_issue_tag: skipped (no transcript_path in hook input)", file=sys.stderr)
        return 0

    try:
        text = last_assistant_text(transcript_path)
    except OSError as e:
        print(f"check_issue_tag: skipped (cannot read transcript: {e})", file=sys.stderr)
        return 0

    if text is None:
        return 0  # no assistant turn yet -- silent, not a violation

    if ISSUE_TAG.match(text) or NONWORK_TAG.match(text):
        return 0  # convention satisfied

    if UNBOUND_TAG.match(text):
        # Satisfied, but this is the billing leak. Say so every time -- that is the point.
        print(
            "check_issue_tag: [unbound] -- WORK was done with no issue to bind it to.\n"
            "  This is the exact leak that produces unbillable/untraceable hours.\n"
            "  File an issue now:  gh issue create --title \"...\"\n"
            "  (If this turn was NOT work, use [status] / [chat] / [meta] instead.)",
            file=sys.stderr,
        )
        return 0

    preview = text.lstrip()[:40].replace("\n", " ")
    print(
        "check_issue_tag: WARNING (#131) - the last assistant turn had NO leading tag.\n"
        "  Every turn must begin with ONE of:\n"
        "    [#123]     work bound to issue 123        <- the normal case\n"
        "    [unbound]  work with no issue yet         <- tripwire: file one\n"
        "    [status]   status/report answer           <- not work, fine\n"
        "    [chat]     conversation/clarification     <- not work, fine\n"
        "    [meta]     quick command, no work product <- not work, fine\n"
        f"  Turn started: {preview!r}...\n"
        "  Untagged turns fall back to inference and weaken time attribution.",
        file=sys.stderr,
    )
    return 0  # advisory: never block


if __name__ == "__main__":
    raise SystemExit(main())
