"""Realchat corpus builder — produces longmem-shaped oracle items from
hand-curated definitions over the realchat session pool.

Input:
  - eval/longmem/data/realchat_sessions.json  (from realchat_extractor.py)
  - REALCHAT_CURATION below                    (hand-curated item definitions)

Output:
  - eval/longmem/data/realchat_oracle.json     (longmem-shaped items)

Item shape — same as longmemeval_oracle.json so the harness runs unchanged:
  {
    "question_id": "rc_001",
    "question_type": "single-session-user" | "multi-session" | "temporal-reasoning" |
                     "knowledge-update" | "single-session-assistant",
    "question": "...",
    "answer": "...",                  # gold; judge does fuzzy match
    "question_date": "YYYY-MM-DD",
    "haystack_dates": ["YYYY-MM-DD", ...],
    "haystack_session_ids": ["...", ...],
    "haystack_sessions": [[{role: ..., content: ...}, ...], [...]]
  }
  (abstention items get question_id ending in "_abs"; same shape.)

Curation philosophy — "diverse axes" per Tom 2026-05-10:
  - 5 axes balanced (3 items each = 15 total).
  - Within an axis, vary topic (recall/encoding/aspects/identity/dampening) and
    time-span (within-session, multi-day, multi-week).
  - Gold answers are qualitative — substring-friendly so the LLM judge has
    room to grant correctness on paraphrases.
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SESSIONS_PATH = ROOT / "eval/longmem/data/realchat_sessions.json"
OUT_PATH = ROOT / "eval/longmem/data/realchat_oracle.json"


# ─── Curation — hand-picked items, axis-balanced ──────────────────────
#
# Each item references session_ids from realchat_sessions.json. For
# multi-session items, list multiple ids. The builder loads those
# sessions and trims `haystack_turns` if specified (default: full session).
#
# Filling rule for `answer`: write what you'd ACCEPT as a correct hypothesis
# from the brain. The judge model checks semantic match, not exact wording.
#
# Filling rule for `question`: write as Tom would actually ask. Don't sanitize
# — naturalness is the test.

REALCHAT_CURATION: List[Dict[str, Any]] = [
    # ─── info_extraction (single-session-user × 3) ───────────────────
    {
        "id": "rc_info_01",
        "axis": "info_extraction",
        "question_type": "single-session-user",
        "session_ids": ["fd829e08-35b9-408b-9ea6-d50cc9e19aec"],
        "haystack_turns_per_session": [20],
        "question": "What variations of longmem evals were we running, and what was the goal?",
        "answer": "We were running multiple longmem variations to test different things — checking we're on the latest version with all components ready, exploring whether there are reasons to believe results will be better. The goal was to validate the longmem evaluation pipeline.",
        "question_date": "2026-04-24",
    },
    {
        "id": "rc_info_02",
        "axis": "info_extraction",
        "question_type": "single-session-user",
        "session_ids": ["8c4cc185-5e00-4b0a-b86e-73f093993b29"],
        "haystack_turns_per_session": [20],
        "question": "What was Tom asking about the embedding matching flow?",
        "answer": "Tom asked for a breakdown of the current flow in embedding matching as well as graph traversing — how the recall pipeline actually works.",
        "question_date": "2026-04-12",
    },
    {
        "id": "rc_info_03",
        "axis": "info_extraction",
        "question_type": "single-session-user",
        "session_ids": ["7c64be2c-a174-4dfb-91b9-aca9f81fff29"],
        "haystack_turns_per_session": [20],
        "question": "Why did Tom say we needed a fresh look on April 30?",
        "answer": "Tom felt we had broken a lot of stuff lately and needed a fresh look — a step-back review rather than continuing to push forward.",
        "question_date": "2026-04-30",
    },
    # ─── multi_session × 3 ───────────────────────────────────────────
    {
        "id": "rc_multi_01",
        "axis": "multi_session",
        "question_type": "multi-session",
        # Three sessions across days about ongoing work
        "session_ids": [
            "8ec23bbe-99d6-42de-87e9-ad2060ed82c7",   # 2026-04-25
            "4516336b-085f-45c2-8b67-e0b157841a96",   # 2026-04-26
            "7c64be2c-a174-4dfb-91b9-aca9f81fff29",   # 2026-04-30
        ],
        "haystack_turns_per_session": [8, 8, 8],
        "question": "What was the multi-day pattern around April 25-30 — what did we work on across those sessions?",
        "answer": "Late April included reviewing prior session results, eval inspection (April 25-26 sessions both started by asking Anchor to recall recent work), and on April 30 Tom flagged that we'd broken a lot of things and needed a fresh look.",
        "question_date": "2026-05-01",
    },
    {
        "id": "rc_multi_02",
        "axis": "multi_session",
        "question_type": "multi-session",
        "session_ids": [
            "f02e85d9-b63e-4406-9807-074115b8e5bc",   # 2026-05-01
            "e3d738b7-6984-40e6-a061-c5f7d3488fba",   # 2026-05-02
            "b4914e53-7af9-492f-8ff1-7bc831b7fd10",   # 2026-05-03
        ],
        "haystack_turns_per_session": [8, 8, 8],
        "question": "What was the energy of early May sessions — how did Tom open them?",
        "answer": "Early May Tom opened with high-energy framings — May 1 'Wakey wakey, lots of work to do', May 2 'Big day today', May 3 'Let's continue Anchor!'. Active forward-momentum framing.",
        "question_date": "2026-05-05",
    },
    {
        "id": "rc_multi_03",
        "axis": "multi_session",
        "question_type": "multi-session",
        "session_ids": [
            "411ca511-14fe-4bb5-aae2-4b55de1c0a45",   # 2026-05-04
            "80eaec64-5bec-4251-9ae0-46551b981a9f",   # 2026-05-05
        ],
        "haystack_turns_per_session": [10, 10],
        "question": "What were the open action items being tracked across these sessions?",
        "answer": "On May 5 Tom opened by asking where we were on all open action items, signaling that action-item tracking was a live concern across that period. The May 4 session was a longer working session feeding into that review.",
        "question_date": "2026-05-06",
    },
    # ─── temporal × 3 ────────────────────────────────────────────────
    {
        "id": "rc_temporal_01",
        "axis": "temporal",
        "question_type": "temporal-reasoning",
        "session_ids": [
            "8ec23bbe-99d6-42de-87e9-ad2060ed82c7",   # 2026-04-25
            "4516336b-085f-45c2-8b67-e0b157841a96",   # 2026-04-26
        ],
        "haystack_turns_per_session": [10, 10],
        "question": "What did we work on the day BEFORE April 26?",
        "answer": "On April 25 we worked on checking what we'd recently encoded and recalled recent memories — the session asked Anchor to verify what was just done.",
        "question_date": "2026-04-27",
    },
    {
        "id": "rc_temporal_02",
        "axis": "temporal",
        "question_type": "temporal-reasoning",
        "session_ids": [
            "5a1dad9f-588a-4227-b581-6483447ec445",   # 2026-04-18
            "5092e709-b5ea-4269-9e8f-959dbedcbaf9",   # 2026-04-17
        ],
        "haystack_turns_per_session": [10, 10],
        "question": "Which session happened earlier — the one where Tom asked to check what was encoded manually, or the 'Heya' opener?",
        "answer": "The 'Heya' opener on 2026-04-17 was earlier; the session where Tom asked Anchor to check what was just encoded manually was on 2026-04-18.",
        "question_date": "2026-04-19",
    },
    {
        "id": "rc_temporal_03",
        "axis": "temporal",
        "question_type": "temporal-reasoning",
        "session_ids": [
            "fd829e08-35b9-408b-9ea6-d50cc9e19aec",   # 2026-04-24
            "8ec23bbe-99d6-42de-87e9-ad2060ed82c7",   # 2026-04-25
            "4516336b-085f-45c2-8b67-e0b157841a96",   # 2026-04-26
        ],
        "haystack_turns_per_session": [8, 8, 8],
        "question": "Tracing a 3-day arc from April 24 to April 26 — what was the through-line?",
        "answer": "Across April 24-26 the through-line was eval validation — April 24 checking longmem variations were running well, April 25 verifying recently-encoded memories, April 26 looking at last session's eval results.",
        "question_date": "2026-04-27",
    },
    # ─── knowledge_update × 3 ────────────────────────────────────────
    {
        "id": "rc_update_01",
        "axis": "knowledge_update",
        "question_type": "knowledge-update",
        "session_ids": ["7c64be2c-a174-4dfb-91b9-aca9f81fff29"],  # 2026-04-30 "we broke a lot"
        "haystack_turns_per_session": [16],
        # Question revised to match verifiable haystack content (was about
        # 'end of session revised view' which my haystack doesn't fully cover)
        "question": "On April 30, after Tom said 'we broke a lot', what was the FIRST concrete problem Anchor identified?",
        "answer": "The hook recall was timing out — Anchor found that the pre-response recall hook had failed with a timeout, meaning Claude wasn't actually getting the recall context. This was identified as a 'live red flag' early in the session.",
        "question_date": "2026-05-01",
    },
    {
        "id": "rc_update_02",
        "axis": "knowledge_update",
        "question_type": "knowledge-update",
        # Replaced — the original "what changed across the whole session" question was
        # too vague to verify within a 16-turn cap. Use a session where Anchor's response
        # ON TURN 0 is a clear correction of the user's framing.
        "session_ids": [
            "f02e85d9-b63e-4406-9807-074115b8e5bc",  # 2026-05-01 — opens with "wakey wakey"
        ],
        "haystack_turns_per_session": [16],
        "question": "What was Tom's ask in early turns, and what did Anchor's response add or correct?",
        "answer": "Tom asked Anchor to pull latest memories and noted there was a doc Anchor had written. Anchor confirmed the state from prior session — three locked anchor nodes plus the handoff doc agreed on what had shipped (ORT arena fix, facts scout v2, hop-3 scrutiny gate) with two open threads remaining.",
        "question_date": "2026-05-02",
    },
    {
        "id": "rc_update_03",
        "axis": "knowledge_update",
        "question_type": "knowledge-update",
        "session_ids": [
            "2b4f31a9-e60e-48e2-8f62-81c4de1ce7eb",  # 2026-04-09 mid-conversation start about S3/interface table
        ],
        "haystack_turns_per_session": [16],
        "question": "What was the initial proposal Tom was confirming, and what got revised?",
        "answer": "Tom was confirming that a prompt should be saved as a knowledge reference in the interaction table — to be revisited and edited by S3 later. The framing positioned the interaction table as the canonical place for evolvable prompts rather than hard-coded values.",
        "question_date": "2026-04-10",
    },
    # ─── abstention × 3 ─────────────────────────────────────────────
    {
        "id": "rc_abs_01_abs",
        "axis": "abstention",
        "question_type": "single-session-user",  # type doesn't matter; _abs suffix routes to abstention axis
        "session_ids": ["fd829e08-35b9-408b-9ea6-d50cc9e19aec"],  # technical eval session
        "haystack_turns_per_session": [16],
        "question": "What is Tom's wife's name?",
        "answer": "The information provided is not enough to answer this question — Tom's wife is not mentioned in any of the haystack sessions.",
        "question_date": "2026-04-25",
    },
    {
        "id": "rc_abs_02_abs",
        "axis": "abstention",
        "question_type": "single-session-user",
        "session_ids": ["8c4cc185-5e00-4b0a-b86e-73f093993b29"],  # embedding flow session
        "haystack_turns_per_session": [16],
        "question": "Did we ever decide to migrate the brain to PostgreSQL?",
        "answer": "The information provided is not enough to answer this question — there is no discussion of migrating to PostgreSQL in the haystack.",
        "question_date": "2026-04-13",
    },
    {
        "id": "rc_abs_03_abs",
        "axis": "abstention",
        "question_type": "multi-session",
        "session_ids": [
            "f02e85d9-b63e-4406-9807-074115b8e5bc",
            "e3d738b7-6984-40e6-a061-c5f7d3488fba",
        ],
        "haystack_turns_per_session": [10, 10],
        "question": "What was the Slack channel name we agreed to use for brain announcements?",
        "answer": "The information provided is not enough to answer this question — no Slack channel for brain announcements is mentioned in any of these sessions.",
        "question_date": "2026-05-03",
    },
]


def build_oracle(sessions: List[Dict[str, Any]],
                 curation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert curation entries into longmem-shaped oracle items."""
    sessions_by_id = {s["session_id"]: s for s in sessions}
    oracle: List[Dict[str, Any]] = []
    misses = []

    for c in curation:
        haystack_sessions = []
        haystack_dates = []
        haystack_session_ids = []
        ok = True
        sids = c["session_ids"]
        truncations = c.get("haystack_turns_per_session") or [None] * len(sids)
        for sid, turn_cap in zip(sids, truncations):
            sess = sessions_by_id.get(sid)
            if not sess:
                misses.append((c["id"], sid))
                ok = False
                break
            exchanges = sess["exchanges"]
            if turn_cap and turn_cap < len(exchanges):
                exchanges = exchanges[:turn_cap]
            haystack_sessions.append(exchanges)
            haystack_dates.append(sess["date"])
            haystack_session_ids.append(sid)
        if not ok:
            continue

        # _abs suffix routes to the abstention axis in the harness regardless
        # of question_type; non-_abs items use question_type for axis.
        oracle.append({
            "question_id": c["id"],
            "question_type": c["question_type"],
            "question": c["question"],
            "answer": c["answer"],
            "question_date": c.get("question_date", ""),
            "haystack_dates": haystack_dates,
            "haystack_session_ids": haystack_session_ids,
            "haystack_sessions": haystack_sessions,
        })

    if misses:
        print("[corpus] WARN missing sessions:", flush=True)
        for item_id, sid in misses:
            print(f"    {item_id} → {sid}", flush=True)
    return oracle


def main():
    if not SESSIONS_PATH.exists():
        print(f"[corpus] {SESSIONS_PATH} missing — run realchat_extractor.py first",
              file=sys.stderr)
        sys.exit(1)
    with open(SESSIONS_PATH) as f:
        sessions = json.load(f)
    print(f"[corpus] loaded {len(sessions)} sessions", flush=True)

    oracle = build_oracle(sessions, REALCHAT_CURATION)
    print(f"[corpus] built {len(oracle)} oracle items "
          f"(curation entries: {len(REALCHAT_CURATION)})", flush=True)

    # Axis distribution
    from collections import Counter
    axes = Counter()
    for item in oracle:
        if item["question_id"].endswith("_abs"):
            axes["abstention"] += 1
        else:
            qt = item["question_type"]
            for axis_name, types in [
                ("info_extraction", ["single-session-user", "single-session-assistant",
                                     "single-session-preference"]),
                ("multi_session", ["multi-session"]),
                ("temporal", ["temporal-reasoning"]),
                ("knowledge_update", ["knowledge-update"]),
            ]:
                if qt in types:
                    axes[axis_name] += 1
                    break
    print(f"[corpus] axis distribution: {dict(axes)}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(oracle, f, indent=2)
    print(f"[corpus] wrote → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
