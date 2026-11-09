"""Scaled benchmark corpus.

The application seeds a small demo knowledge base (``app/retrieval/corpus.py``).
That is the right size for a demo and the wrong size for evaluation: with 56
documents recall@5 sits close to its ceiling, so every retriever looks good and
the hybrid gain is compressed into a few percentage points of headroom.

This module builds a larger evaluation corpus from the same 20 primary documents
plus generated distractors. Two kinds of distractor matter:

* **Bulk** - plausible platform runbooks that make the candidate pool large.
* **Hard negatives** - documents that share vocabulary with a primary document
  but do not answer its questions: deprecated v0 variants, adjacent products,
  and near-miss identifiers (``CUE-E1041`` beside ``CUE-E1042``).

Hard negatives are what make a retrieval benchmark non-trivial. Without them a
larger corpus mostly adds documents that nobody would ever confuse with the
answer.

Everything here is deterministic: same corpus on every machine, no sampling.
"""

from __future__ import annotations

from typing import Any

from app.retrieval.corpus import DOCUMENTS as PRIMARY_DOCUMENTS

# ── hard negatives ─────────────────────────────────────────────────────────
# Each of these is deliberately confusable with one or more primary documents
# while answering a *different* question. None of them contains an answer that
# any labelled query expects.
HARD_NEGATIVES: list[dict[str, Any]] = [
    {
        "doc_id": "kb-neg-legacy-protocol",
        "title": "Legacy v0 gateway protocol (deprecated)",
        "tags": ["deprecated", "protocol"],
        "text": """
The v0 protocol used a different frame vocabulary and was removed in gateway v1.
Where v1 sends `ack`, `retrieval` and `tool_call_start`, v0 sent a single
`chunk` frame with a numeric opcode and a trailing `end` frame. There was no
per-turn sequence number, so a v0 client could not detect a dropped frame after a
reconnect and had to re-render the whole answer.

v0 had no `hello` handshake: the first frame was always a `prompt`. Authentication
was a plaintext token in the query string with no expiry, which is why the v0
endpoint is disabled by default and only reachable when
legacy_protocol_enabled=true.

Migration notes for anyone still running a v0 client: map `chunk` to `delta`,
`end` to `done`, and drop the opcode field. The old client libraries under
sdk/legacy are unmaintained and are not part of the v1 compatibility promise.
""",
    },
    {
        "doc_id": "kb-neg-mobile-reconnect",
        "title": "Mobile SDK reconnect behaviour",
        "tags": ["mobile", "resilience"],
        "text": """
The mobile SDK reconnects more conservatively than the desktop client because
radio wake-ups are expensive and battery drain is a user-visible bug. Its base
delay is 500 milliseconds and its cap is 30 seconds, with decorrelated jitter
rather than full jitter so that a fleet waking from background does not converge
on the same schedule.

On iOS the SDK suspends the socket when the app is backgrounded and reconnects on
foreground. On Android it uses a WorkManager job with a minimum latency of 15
seconds. Neither platform retries while the device reports no network, because a
failed attempt would only add latency to the eventual success.

The mobile SDK does not resume conversation history by default. Sessions are
treated as disposable and the UI re-renders from the local cache, which keeps the
handshake small on metered connections.
""",
    },
    {
        "doc_id": "kb-neg-relevance-experiments",
        "title": "Search relevance experiments, Q3 summary",
        "tags": ["experiments", "search"],
        "text": """
This quarter we ran four offline relevance experiments. The headline result is
that a learned reranker improved nDCG@10 by 4.1 points over the current production
ranking, at the cost of roughly 40 milliseconds of added query latency.

Experiment 2 replaced the lexical scorer with a neural retriever and improved
paraphrase recall but regressed on identifier queries, which matches the
well-documented weakness of dense models on rare tokens. Experiment 3 combined
both candidate lists and was the only configuration that beat the incumbent on
every slice we track.

Experiment 4 tested query expansion with an LLM and was rejected: it improved
long-tail recall slightly but made latency unpredictable and produced occasional
hallucinated expansions that hurt precision. The experiment log and per-slice
numbers live in the search-experiments workspace.
""",
    },
    {
        "doc_id": "kb-neg-vector-bakeoff",
        "title": "Vector database bake-off",
        "tags": ["evaluation", "infra"],
        "text": """
We compared four vector databases on a 2 million vector corpus with 768
dimensions. Qdrant and Milvus both support HNSW with a tunable ef_search at query
time; Weaviate exposes a similar knob under a different name; the managed offering
we tested hides it entirely behind a latency tier.

Recall at 10 ranged from 0.91 to 0.97 depending on the ef_search setting and the
amount of quantization applied. Scalar quantization to int8 cut memory by roughly
70 percent and cost about two points of recall in the worst configuration.

Operationally the differentiator was not raw speed but how each system handled
index rebuilds during ingest. Two of the four required a full rebuild, which is a
non-starter for a corpus that changes hourly. The recommendation memo is in the
infra-decisions folder and is unrelated to the pgvector deployment used by the
gateway.
""",
    },
    {
        "doc_id": "kb-neg-transcription-vendor",
        "title": "Contact centre transcription vendor review",
        "tags": ["procurement", "audio"],
        "text": """
The contact centre team evaluated three transcription vendors for call recording
compliance. All three support 8 kHz telephony audio and diarisation; only one
supports real-time partial results over a websocket.

Word error rate on the internal call set ranged from 11 to 18 percent, with the
gap driven almost entirely by accented speech and background noise. None of the
vendors met the 5 percent target the compliance team asked for, so the
recommendation is to keep human review for flagged calls.

Pricing is per audio minute with a minimum commit. The review is scoped to the
contact centre product and does not cover the desktop assistant's dictation
feature, which uses a different vendor and a different audio pipeline.
""",
    },
    {
        "doc_id": "kb-neg-desktop-budget-legacy",
        "title": "Desktop performance budget (Electron 12, retired)",
        "tags": ["deprecated", "performance"],
        "text": """
The retired Electron 12 client had a 16 millisecond frame budget and a 120
millisecond interaction budget. We measured long tasks with the Performance
Observer API and tracked cumulative layout shift on the transcript view.

The old client re-rendered the entire transcript on every token, which was
acceptable at the 20 token/second streams of the time and became the top
complaint once models started streaming faster. The fix at the time was to
throttle rendering with a 100 millisecond timer, which introduced visible lag but
kept the main thread responsive.

This document is kept for historical context. It predates the current client's
animation-frame batching and its memory guidance, and the numbers should not be
used for capacity planning.
""",
    },
    {
        "doc_id": "kb-neg-cost-governance",
        "title": "Cost governance handbook",
        "tags": ["finance", "governance"],
        "text": """
Every team owns a monthly cloud budget and is expected to review it in the second
week. The finance team publishes a showback report grouped by service and
environment, with a one-week lag. Variances above 15 percent require a written
explanation in the cost channel.

The handbook covers tagging standards, commitment purchasing, and the exception
process for experimental workloads. It does not cover per-request pricing of
third-party model providers; that is handled by the product teams that own the
integration, using the rate cards in the procurement system.

Chargeback is calculated monthly from the showback report. Teams that consistently
overshoot are paired with a finance partner rather than being blocked, because the
goal is forecasting accuracy rather than punishment.
""",
    },
    {
        "doc_id": "kb-neg-provider-outage",
        "title": "Incident runbook: model provider outage",
        "tags": ["incident", "providers"],
        "text": """
When a model provider reports a widespread outage, the on-call engineer first
confirms the scope on the provider status page, then checks whether the failure is
affecting one region or all of them.

The immediate mitigation is to route traffic to the secondary provider through the
routing layer. This is a manual switch because automatic failover during a partial
outage can move traffic onto a provider that is also degraded. After switching,
watch the error rate for ten minutes before declaring mitigation.

Communication follows the standard severity process: an incident channel, a status
page update within fifteen minutes, and a customer-facing note if the outage
exceeds thirty minutes. The postmortem is due within five business days and focuses
on detection time rather than the provider's root cause, which we do not control.
""",
    },
    {
        "doc_id": "kb-neg-error-index-v0",
        "title": "Error code index (v0 gateway)",
        "tags": ["deprecated", "errors"],
        "text": """
This index lists error codes from the retired v0 gateway. Codes are retained so
that historical support tickets remain searchable.

CUE-E1041 indicated a malformed prompt payload in v0; it was replaced by the
stricter frame validation introduced in v1. CUE-E1043 was an internal assertion
failure in the v0 coalescer and no longer exists.

CUE-E2040 was the v0 name for a provider authentication failure. CUE-E2042 covered
quota exhaustion on the old free tier, which was discontinued. CUE-E5031 was a tool
registry lookup failure before the registry became schema-validated, and CUE-E5033
was a duplicate tool registration warning.

No code in this index is emitted by the current gateway. Consult the v1 error
reference instead.
""",
    },
    {
        "doc_id": "kb-neg-audio-codec",
        "title": "Audio codec evaluation for call recording",
        "tags": ["audio", "evaluation"],
        "text": """
We evaluated Opus, AAC and raw PCM for call recording storage. Opus at 24 kbps
gave transparent quality for speech and cut storage by roughly 90 percent compared
with 16-bit PCM, which is the dominant cost in a long-retention archive.

The trade-off is decode cost on the analytics path: every downstream job has to
decode before it can run voice activity detection, and that decode shows up in job
latency. For recordings that are transcribed once and then archived, the storage
saving wins comfortably.

This evaluation covers archival storage only. Live capture keeps raw frames in
memory because encoding adds a buffering stage that hurts caption latency, and the
capture path deliberately has no codec dependency.
""",
    },
    {
        "doc_id": "kb-neg-prompt-guide",
        "title": "Prompt engineering guide for support macros",
        "tags": ["prompting", "support"],
        "text": """
Support macros are short prompts that turn a ticket summary into a customer-facing
reply. The guide recommends a fixed structure: restate the customer's problem in
one sentence, give the resolution, then list anything the customer must do.

Keep macros under 200 words. Longer prompts drift toward hedging language and
produce replies that read as if they were written by a committee. Avoid promising
timelines unless the macro pulls the date from the ticket fields.

Every macro is reviewed by a support lead before it ships, and macros are versioned
so that a regression can be traced to a specific edit. The guide does not cover
model selection or retrieval; those are platform concerns handled elsewhere.
""",
    },
    {
        "doc_id": "kb-neg-autoscaling",
        "title": "Kubernetes autoscaling policy",
        "tags": ["infra", "kubernetes"],
        "text": """
Services scale on a combination of CPU utilisation and a custom queue-depth metric.
The horizontal pod autoscaler targets 60 percent CPU, with a scale-up window of 30
seconds and a scale-down window of five minutes. Scale-down is deliberately slow so
that a brief dip does not cause a thrash cycle.

Each deployment declares a pod disruption budget of at least two replicas, and
cluster autoscaling adds nodes when pods stay pending for more than two minutes.
Node pools are separated by workload class so that a batch job cannot starve
latency-sensitive services.

This policy applies to the platform cluster. The gateway deployment runs on a
separate cluster with its own policy because long-lived websocket connections make
connection-draining behaviour more important than pod churn speed.
""",
    },
]


# ── bulk distractors ───────────────────────────────────────────────────────
# (family, slug, subtopics, keywords)
FAMILIES: list[tuple[str, str, list[str], list[str]]] = [
    (
        "billing invoices",
        "billing",
        ["proration", "tax identifiers", "dunning", "credit notes", "multi-currency"],
        ["invoice", "payment method", "settlement", "refund"],
    ),
    (
        "workspace onboarding",
        "onboarding",
        ["invite flows", "SSO setup", "role assignment", "domain verification"],
        ["workspace", "teammate", "provisioning", "seat"],
    ),
    (
        "marketing attribution",
        "attribution",
        ["UTM handling", "conversion windows", "campaign identifiers", "channel overlap"],
        ["campaign", "touchpoint", "conversion", "channel"],
    ),
    (
        "mobile offline mode",
        "mobile",
        ["local cache", "conflict resolution", "sync queue", "background refresh"],
        ["offline", "device", "sync", "cache"],
    ),
    (
        "data export",
        "export",
        ["CSV export", "NDJSON streaming", "signed download URLs", "retention windows"],
        ["export", "download", "archive", "retention"],
    ),
    (
        "webhook delivery",
        "webhook",
        ["signature headers", "retry schedule", "dead letter queue", "endpoint rotation"],
        ["webhook", "delivery", "subscriber", "payload"],
    ),
    (
        "search analytics",
        "analytics",
        ["query volume", "zero-result rate", "click-through", "session stitching"],
        ["analytics", "query", "metric", "dashboard"],
    ),
    (
        "accessibility",
        "a11y",
        ["focus order", "ARIA labelling", "reduced motion", "contrast auditing"],
        ["accessibility", "screen reader", "keyboard", "contrast"],
    ),
    (
        "localization",
        "i18n",
        ["locale files", "plural rules", "right-to-left layout", "translation memory"],
        ["locale", "translation", "language", "string"],
    ),
    (
        "email deliverability",
        "email",
        ["SPF records", "DKIM signing", "DMARC policy", "bounce classification"],
        ["email", "delivery", "bounce", "sender"],
    ),
    (
        "feature flags",
        "flags",
        ["percentage rollout", "targeting rules", "kill switches", "flag lifecycle"],
        ["flag", "rollout", "targeting", "experiment"],
    ),
    (
        "audit logging",
        "audit",
        ["actor attribution", "immutable retention", "log export", "tamper evidence"],
        ["audit", "log", "actor", "retention"],
    ),
    (
        "backup and restore",
        "backup",
        ["snapshot cadence", "point-in-time recovery", "restore drills", "offsite copies"],
        ["backup", "restore", "snapshot", "recovery"],
    ),
    (
        "incident response",
        "incident",
        ["severity levels", "on-call rotation", "postmortem template", "comms cadence"],
        ["incident", "severity", "on-call", "postmortem"],
    ),
    (
        "capacity planning",
        "capacity",
        ["headroom targets", "p99 load modelling", "autoscaling limits", "growth forecast"],
        ["capacity", "headroom", "load", "forecast"],
    ),
    (
        "api versioning",
        "versioning",
        ["deprecation windows", "sunset headers", "changelog discipline", "client pinning"],
        ["version", "deprecation", "client", "changelog"],
    ),
    (
        "key rotation",
        "keys",
        ["rotation cadence", "dual-key overlap", "revocation", "envelope encryption"],
        ["key", "rotation", "secret", "encryption"],
    ),
    (
        "file storage",
        "storage",
        ["presigned uploads", "virus scanning", "lifecycle rules", "object versioning"],
        ["storage", "upload", "object", "bucket"],
    ),
    (
        "tenant isolation",
        "tenancy",
        ["row-level security", "noisy neighbour", "per-tenant quotas", "data boundaries"],
        ["tenant", "isolation", "quota", "workspace"],
    ),
]


_TEMPLATES = [
    """{title}

This runbook covers {family} for the platform team, with emphasis on {subtopic}.

Ownership and escalation
The primary owner rotates quarterly. Escalate to the platform on-call when a {slug} change is expected to affect more than five percent of traffic, or when the workspace has already burned half of its monthly error budget.

Operational notes
Review the {slug} dashboard weekly and keep the documented thresholds current. Every exception is recorded with a link to the change ticket, and dashboards are grouped by workspace and region so that a single noisy tenant is visible immediately.

Change procedure
Open a change ticket, request review from the owning team, roll out to the staging workspace first, and watch error budget burn for one hour before promoting to production. Rollback is the previous known-good configuration, which is retained for thirty days.""",
    """{title}

Background
The {family} surface grew from a single script into a service with several consumers across multiple regions. This cycle's {subtopic} work focuses on making that growth boring: fewer manual steps, clearer ownership, and a rollback that any engineer on the team can execute without asking questions.

What changed
The {slug} pipeline now validates its inputs before doing any work, emits a structured event for every state transition, and records the configuration hash alongside each result so that a regression can be tied to a specific deploy.

Verification
Run the {slug} smoke check against staging and confirm that the expected {keyword} records appear within one minute. A green check with no records is treated as a failure, because it usually means the consumer is silently dropping events.

Rollback
Restore the previous configuration snapshot, restart the consumer, and replay the last hour of events from the durable queue. Replay is idempotent.""",
    """{title}

Why this exists
Support tickets about {family} were arriving with no shared vocabulary, so the first twenty minutes of every investigation went into working out which system the reporter meant. This document is the shared reference for {subtopic}.

How it works
The {slug} service owns the write path and publishes a change event after the database commit, never before. Readers subscribe to the event stream and maintain their own projections, which means a slow reader cannot block a writer. Projections are rebuilt from the event log when a consumer ships a bug.

Failure behaviour
If the {slug} consumer falls behind by more than ten minutes it sheds load rather than queueing indefinitely, and the {keyword} dashboard turns amber. Falling behind is recoverable; serving stale data without a signal is not.

What this is not
This document does not cover pricing, contracts, or the customer-facing explanation of {family}. Those live with the owning product team.""",
    """{title}

Scope
Applies to every environment that runs the {family} workload, including the staging workspace used for release rehearsal.

Configuration
The {slug} settings are managed declaratively and reviewed like code. The two values that matter most are the batch size, which trades throughput against tail latency, and the timeout, which should be set from the observed p99 rather than from intuition. Both are documented next to the deploy manifest.

Monitoring
We alert on symptoms rather than causes: elevated error rate, exhausted {keyword} capacity, and a queue that is not draining. Cause-based alerts were removed because they fired for conditions that resolved themselves and trained the team to ignore the page.

Review cadence
This runbook is reviewed each quarter and after every incident that touches {subtopic}. The review is a fifteen minute read-through, not a rewrite; if it takes longer than that, the document has drifted too far from reality.""",
    """{title}

Summary
{subtopic_cap} is the part of {family} that most often surprises new engineers, because the obvious implementation is subtly wrong in a way that only shows up under load.

Details
The {slug} path buffers writes for a short window before flushing, which keeps the {keyword} backend from seeing one request per event. The window is bounded in both time and size: whichever limit is hit first triggers the flush. A window that is bounded only in time will melt the backend during a burst, and a window bounded only in size adds latency to quiet traffic.

Gotchas
Retries must be idempotent, so every write carries a client-generated key and the backend deduplicates on it. Without that key a retry after a timeout creates a duplicate record, and the duplicate is invisible until someone reconciles totals.

Related
See the {family} overview for how this fits into the wider platform, and the incident runbook for what to do when the {keyword} backend is unavailable.""",
]


def _bulk_documents(count: int) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    per_family = max(1, count // len(FAMILIES) + 1)
    index = 0
    for family, slug, subtopics, keywords in FAMILIES:
        for offset in range(per_family):
            if len(docs) >= count:
                return docs
            subtopic = subtopics[offset % len(subtopics)]
            keyword = keywords[offset % len(keywords)]
            template = _TEMPLATES[index % len(_TEMPLATES)]
            docs.append(
                {
                    "doc_id": f"kb-{slug}-{offset:02d}",
                    "title": f"{family.title()}: {subtopic} ({offset:02d})",
                    "tags": ["distractor", slug],
                    "text": template.format(
                        title=f"{family.title()}: {subtopic}",
                        family=family,
                        slug=slug,
                        subtopic=subtopic,
                        subtopic_cap=subtopic[0].upper() + subtopic[1:],
                        keyword=keyword,
                    ),
                }
            )
            index += 1
    return docs


def benchmark_documents(total: int = 412) -> list[dict[str, Any]]:
    """Primary documents + hard negatives + bulk fill, exactly `total` docs."""
    docs = list(PRIMARY_DOCUMENTS) + list(HARD_NEGATIVES)
    remaining = max(0, total - len(docs))
    docs.extend(_bulk_documents(remaining))
    return docs


def primary_ids() -> set[str]:
    return {doc["doc_id"] for doc in PRIMARY_DOCUMENTS}


def distractor_ids() -> set[str]:
    return {doc["doc_id"] for doc in benchmark_documents() if doc["doc_id"] not in primary_ids()}
