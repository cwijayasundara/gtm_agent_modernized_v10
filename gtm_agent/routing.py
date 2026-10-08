"""Query-based model routing for the GTM agent, via `decision_harness`.

Four LLM options in three tiers (catalog overlay in `catalog_overlay.json`):

  fast        glm-5.3-flash  (fireworks)   - mechanical steps, simple lookups
  fast alt    deepseek-4-flash (fireworks)  - fast-tier alternate
  balanced    gpt-6-luna      (openai)      - ordinary multi-step work
  performance claude-haiku-5-5 (anthropic)  - drafting, ambiguous, escalations

How the decision is made:
- With DECISIONS_API_KEY / OPENAI_API_KEY set, the OpenAI Decisions API
  classifies each thread's first query (tier choice + confidence, complexity,
  planning predicate) in ONE fast typed-questions request.
- Without it, `decision_harness` degrades to its deterministic offline
  heuristic classifier — same routing shape, zero network, zero cost.

The chosen model is sticky per thread (prompt-cache friendly), escalates to
the next tier on provider failure, and every call lands in a `CostLedger`
with tokens, $ cost (catalog prices), and response time. A summary report is
printed and the full ledger is exported to `out/usage/`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain.agents.middleware.types import AgentMiddleware

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from . import tools
from .agent import PROJECT_ROOT, GTM_SYSTEM_PROMPT

OUT_DIR = Path(__file__).resolve().parent.parent / "out"
OVERLAY = Path(__file__).resolve().parent / "catalog_overlay.json"

# Labels of the models the GTM agent may run on (provider/id, per catalog).
AVAILABLE_LABELS = [
    "fireworks/accounts/fireworks/routers/glm-5p3-fast",
    "fireworks/accounts/fireworks/models/deepseek-4-flash",
    "openai/gpt-6-luna",
    "anthropic/claude-haiku-5-5",
]

FRIENDLY_NAMES = {
    "glm-5p3-fast": "glm-5.3-flash (fireworks)",
    "glm-5p3-flash": "glm-5.3-flash (fireworks)",
    "deepseek-4-flash": "deepseek-4-flash (fireworks)",
    "gpt-6-luna": "gpt-6-luna (openai)",
    "claude-haiku-5-5": "claude-haiku-5-5 (anthropic)",
}


def friendly(label: str) -> str:
    return FRIENDLY_NAMES.get(label.split("/")[-1], label)


class CallTelemetryMiddleware(AgentMiddleware):
    """Times every model call and captures token details incl. prompt-cache
    hits (outermost middleware, so retry/escalation time is included)."""

    name = "call_telemetry"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        started = time.perf_counter()
        response = handler(request)
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage: dict[str, Any] = {}
        try:
            for message in reversed(response.result if hasattr(response, "result") else response.messages):
                meta = getattr(message, "usage_metadata", None)
                if meta:
                    usage = {
                        "input_tokens": meta.get("input_tokens", 0),
                        "output_tokens": meta.get("output_tokens", 0),
                        "cache_read": (meta.get("input_token_details") or {}).get("cache_read", 0) or 0,
                        "cache_creation": (meta.get("input_token_details") or {}).get("cache_creation", 0) or 0,
                    }
                    break
        except (TypeError, AttributeError):
            pass
        self.calls.append({"latency_ms": latency_ms, **usage})
        return response


class RoutingTelemetry:
    """Collects routing decisions + the cost ledger, prints and saves a report."""

    def __init__(self) -> None:
        self.decisions: list[dict[str, Any]] = []
        self.ledger: Any | None = None  # wired to the middleware's CostLedger
        self.calls_middleware: Any | None = None  # wired to the CallTelemetryMiddleware

    def on_decision(self, decision: Any) -> None:
        c = getattr(decision, "classification", None)
        entry = {
            "model": decision.model.label(),
            "tier": decision.tier.value,
            "reason": decision.reason,
            "sticky": getattr(decision, "sticky", False),
            "tier_confidence": getattr(c, "tier_confidence", None),
            "complexity": getattr(c, "complexity", None),
            "needs_planning": getattr(c, "needs_planning", None),
            "classifier": getattr(c, "classifier", "") if c else "sticky",
            "classification_latency_ms": getattr(c, "latency_ms", 0) if c else 0,
            "events": list(decision.events),
        }
        # The middleware fires this hook twice on sticky calls (decide + post-record);
        # keep consecutive duplicates out of the report.
        if self.decisions and self.decisions[-1] == entry:
            return
        self.decisions.append(entry)

    # -- reporting ------------------------------------------------------------

    def print_report(self) -> Path | None:
        print("\n📊 Routing & usage report")
        print("-" * 64)
        classified = [d for d in self.decisions if not d["sticky"]]
        for d in classified:
            conf = d["tier_confidence"]
            conf_s = f"{conf:.2f}" if isinstance(conf, float) else "—"
            print(
                f"  routed: tier={d['tier']:<11} conf={conf_s} "
                f"complexity={d['complexity']} → {friendly(d['model'])} "
                f"[{d['reason']} via {d['classifier'] or 'sticky'}]"
            )
        sticky_by_model: dict[str, int] = {}
        if self.ledger is not None:
            for e in self.ledger.entries:  # one entry per call; the reason is authoritative
                if e.reason == "sticky":
                    label = friendly(f"{e.provider}/{e.model}")
                    sticky_by_model[label] = sticky_by_model.get(label, 0) + 1
        for label, n in sorted(sticky_by_model.items()):
            print(f"  sticky: {n:>3} follow-up calls stayed on {label} (prompt cache warm)")
        if self.calls_middleware and self.calls_middleware.calls:
            total_ms = sum(c["latency_ms"] for c in self.calls_middleware.calls)
            cache_read = sum(c.get("cache_read", 0) for c in self.calls_middleware.calls)
            cache_write = sum(c.get("cache_creation", 0) for c in self.calls_middleware.calls)
            plain = sum(c.get("input_tokens", 0) for c in self.calls_middleware.calls) - cache_read - cache_write
            print(
                f"  latency: {len(self.calls_middleware.calls)} model calls, "
                f"{total_ms}ms total model time | prompt cache: {cache_read} read / "
                f"{cache_write} written / {plain} uncached input tokens"
            )
        if self.ledger is None or not self.ledger.entries:
            print("  (no billed calls recorded)")
            return None

        entries = self.ledger.entries
        print("-" * 64)
        print(f"  {'model':<38} {'calls':>5} {'in-tok':>8} {'out-tok':>8} {'cost':>9} {'avg ms':>8}")
        by_model: dict[str, list[Any]] = {}
        for e in entries:
            by_model.setdefault(f"{e.provider}/{e.model}", []).append(e)
        for label, rows in sorted(by_model.items()):
            avg = sum(r.latency_ms for r in rows) // max(len(rows), 1)
            cost = sum(r.cost for r in rows)
            it = sum(r.input_tokens for r in rows)
            ot = sum(r.output_tokens for r in rows)
            print(f"  {friendly(label):<38} {len(rows):>5} {it:>8} {ot:>8} ${cost:>8.4f} {avg:>8}")
        grand = self.ledger.totals()["grand"]
        print("-" * 64)
        print(
            f"  total: {grand['calls']} calls, "
            f"{grand['input_tokens']} in / {grand['output_tokens']} out tokens, "
            f"${grand['cost']:.4f}"
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / "usage" / f"ledger-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger.to_json(path)
        print(f"  ledger saved: {path}")
        return path


# ---------------------------------------------------------------------------
# Agent builders
# ---------------------------------------------------------------------------


def _prune_to_credentialed(catalog: Any) -> Any:
    """Drop catalog entries whose provider API key isn't really configured, so
    live runs never construct a model we can't call — the escalation policy
    handles the resulting empty tiers gracefully ("no eligible model →
    escalating tier").

    A key counts only if it looks real (non-empty, >= 20 chars). Providers can
    also be force-restricted via `GTM_ALLOWED_PROVIDERS` (comma-separated),
    e.g. `GTM_ALLOWED_PROVIDERS=anthropic` to skip a broken key and the
    wasted failed round trip it would cause.
    """
    import os

    from decision_harness import ModelCatalog

    allowed = {
        p.strip().lower()
        for p in os.getenv("GTM_ALLOWED_PROVIDERS", "").split(",")
        if p.strip()
    }

    def _usable(spec: Any) -> bool:
        if allowed and spec.provider.lower() not in allowed:
            return False
        if not spec.api_key_env:
            return True
        key = os.getenv(spec.api_key_env, "")
        return len(key) >= 20

    keep = [s for s in catalog.all() if _usable(s)]
    return ModelCatalog(keep) if keep else catalog


def build_selector(use_decisions: bool | None = None, *, prune: bool = False):
    """ModelSelector over the GTM catalog overlay.

    use_decisions=True forces the Decisions API client (needs a key);
    None auto-detects: with a key use it, otherwise pure offline heuristic.
    """
    import os

    from decision_harness import DecisionsClient, ModelCatalog, ModelSelector, Tier
    from decision_harness.heuristic import OfflineFallbackClassifier

    class GTMFallbackClassifier(OfflineFallbackClassifier):
        """The stock fallback scores engineering vocabulary; this adds GTM
        signals so offline routing still meaningfully differentiates the
        agent's workflows (draft vs briefing vs strategy review)."""

        name = "heuristic+gtm"

        _GTM_FAST = (
            "lookup", "list", "inbox", "show", "when is", "status of",
            "open tickets", "simple", "quick check", "date",
        )
        _GTM_BALANCED = (
            "draft", "email", "follow-up", "follow up", "outreach",
            "briefing", "summarize", "meeting prep", "prep", "research",
            "process", "run the standard", "inbound lead",
        )
        _GTM_PERFORMANCE = (
            "strategy", "positioning", "renewal", "churn", "deal risk",
            "competitive", "expansion plan", "escalation", "pricing strategy",
            "quarterly business review", "qbr", "cross-team", "high-consequence",
        )

        def classify(self, task: str):  # noqa: ANN201 - matches parent
            text = task.lower()
            gt_fast = sum(1 for s in self._GTM_FAST if s in text)
            gt_bal = sum(1 for s in self._GTM_BALANCED if s in text)
            gt_perf = sum(1 for s in self._GTM_PERFORMANCE if s in text)
            classification = super().classify(task)
            # GTM vocabulary dominates when it fires at all.
            if gt_perf > 0 and gt_perf >= max(gt_fast, gt_bal):
                classification.tier = Tier.PERFORMANCE
                classification.tier_confidence = 0.72
            elif gt_bal > gt_fast:
                classification.tier = Tier.BALANCED
                classification.tier_confidence = 0.7
            elif gt_fast > 0 or gt_bal > 0 or gt_perf > 0:
                classification.tier = Tier.FAST
            classification.complexity = round(
                min(3.0, classification.complexity + gt_perf * 1.2 + gt_bal * 0.3), 2
            )
            classification.needs_planning = round(
                min(1.0, classification.needs_planning + gt_perf * 0.3), 2
            )
            classification.raw["gtm_signals"] = {"fast": gt_fast, "balanced": gt_bal, "performance": gt_perf}
            return classification

    catalog = ModelCatalog.load(path=OVERLAY)
    if prune:
        catalog = _prune_to_credentialed(catalog)
    client = None
    if use_decisions is None:
        use_decisions = bool(os.getenv("DECISIONS_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if use_decisions:
        client = DecisionsClient()
    selector = ModelSelector(client=client, catalog=catalog)
    selector._fallback = GTMFallbackClassifier()
    return selector


def build_routed_agent(
    telemetry: RoutingTelemetry | None = None,
    *,
    use_decisions: bool | None = None,
    models_override: dict[str, Any] | None = None,
):
    """GTM deep agent whose model is chosen per query (sticky per thread).

    models_override maps catalog labels (`provider/id`) to chat model
    instances — how offline runs inject the scripted fake for every tier.
    """
    from decision_harness.middleware.model_selection import ModelSelectionMiddleware

    telemetry = telemetry or RoutingTelemetry()
    # Live runs (no injected models) route only to providers we have keys for;
    # offline runs keep the whole catalog since fakes are injected per label.
    selector = build_selector(use_decisions=use_decisions, prune=models_override is None)
    telemetry.ledger = None  # wired to the middleware instance below

    middleware = ModelSelectionMiddleware(
        selector=selector,
        models=models_override,
        on_decision=telemetry.on_decision,
        fallback_model="anthropic:claude-haiku-5-5",
    )
    telemetry.ledger = middleware.ledger
    telemetry.calls_middleware = CallTelemetryMiddleware()

    import os

    nominal = ChatAnthropic(
        model="claude-haiku-5-5",
        max_tokens=8000,
        # Offline runs never call this model (the middleware swaps it), but
        # construction still validates a key — dummy one if none is configured.
        **({} if os.getenv("ANTHROPIC_API_KEY") else {"api_key": "unused-offline"}),
    )
    agent = create_deep_agent(
        # Nominal model; every call is swapped by the routing middleware.
        model=nominal,
        tools=tools.ALL_TOOLS,
        system_prompt=GTM_SYSTEM_PROMPT,
        skills=["/skills/"],
        backend=FilesystemBackend(root_dir=PROJECT_ROOT),
        # Telemetry outermost: measures the full per-call time incl. retries.
        middleware=[telemetry.calls_middleware, middleware],
    )
    return agent


def classify_query(query: str, *, use_decisions: bool | None = None) -> dict[str, Any]:
    """Classify a query and return the routing decision (no agent run). Works offline."""
    selector = build_selector(use_decisions=use_decisions)
    decision = selector.select(query)
    c = decision.classification
    return {
        "query": query,
        "chosen_model": decision.model.label(),
        "tier": decision.tier.value,
        "reason": decision.reason,
        "tier_confidence": c.tier_confidence if c else None,
        "complexity": c.complexity if c else None,
        "needs_planning": c.needs_planning if c else None,
        "classifier": c.classifier if c else "sticky",
        "classification_latency_ms": c.latency_ms if c else 0,
        "events": list(decision.events),
    }
