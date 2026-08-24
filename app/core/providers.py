from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import httpx

from .config import Settings
from .errors import ProviderError, RetryableError
from .retry import call as retry_call
from .trace import Trace

# --------------------------------------------------------------------------
# Wire types. These are ours, not any vendor's -- swapping providers must not
# ripple into application code.
# --------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls("assistant", content, tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> "Message":
        return cls("tool", content, tool_call_id=tool_call_id)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"


class LLMClient(Protocol):
    def chat(self, messages: Sequence[Message], *, tools: Sequence[ToolSpec] | None = None,
             temperature: float | None = None, max_tokens: int | None = None) -> LLMResponse: ...


class EmbeddingClient(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# --------------------------------------------------------------------------
# Pricing. USD per 1M tokens. Update when you re-quote a client -- these
# numbers end up in the cost report you hand them.
# --------------------------------------------------------------------------
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1":               (2.00, 8.00),
    "gpt-4.1-mini":          (0.40, 1.60),
    "gpt-4o":                (2.50, 10.00),
    "gpt-4o-mini":           (0.15, 0.60),
    "claude-sonnet-4-5":     (3.00, 15.00),
    "claude-haiku-4-5":      (1.00, 5.00),
    "claude-opus-4-1":       (15.00, 75.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    # OpenRouter model ids carry a vendor prefix and are billed at the
    # underlying provider's own rate (OpenRouter adds no markup) -- priced
    # here as their own entries rather than trying to strip the prefix and
    # reuse the row above, since that's one more place a typo silently zeros
    # out cost tracking.
    "openai/gpt-4o-mini":     (0.15, 0.60),
    "qwen/qwen3-embedding-8b": (0.01, 0.0),
}


def price(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICING.get(model, (0.0, 0.0))
    return round((tokens_in * p_in + tokens_out * p_out) / 1_000_000, 6)


def approx_tokens(text: str) -> int:
    """~4 chars per token. Good enough for budgeting, never for billing."""
    return max(1, math.ceil(len(text) / 4))


# --------------------------------------------------------------------------
# Mock provider -- the whole system runs offline against this.
# --------------------------------------------------------------------------


class MockLLM:
    """Deterministic fake model.

    Two reasons this exists and is a first-class provider rather than a test
    fixture: CI runs the real code paths with no API key and no spend, and you
    can demo the architecture to a client before they have given you a key.

    Behaviour is scriptable: pass `scripted=[...]` to make it return specific
    responses in order (used by the agent tests to drive tool-calling loops).
    """

    def __init__(self, model: str = "mock", scripted: Sequence[LLMResponse] | None = None):
        self.model = model
        self.scripted = list(scripted or [])
        self.calls: list[list[Message]] = []

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> LLMResponse:
        msgs = list(messages)
        self.calls.append(msgs)
        if self.scripted:
            return self.scripted.pop(0)

        last_user = next((m.content for m in reversed(msgs) if m.role == "user"), "")
        tin = sum(approx_tokens(m.content) for m in msgs)

        # If tools are on offer and none has been run yet, pick one by keyword
        # overlap. Crude, but it means the agent loop, the tool-result path and
        # the approval gate all execute offline instead of being skipped -- which
        # is the whole point of having a mock provider in the first place.
        if tools and not any(m.role == "tool" for m in msgs):
            chosen = _best_tool(last_user, tools)
            if chosen is not None:
                args = _mock_arguments(chosen, last_user)
                call = ToolCall(id=f"mockcall_{len(self.calls)}", name=chosen.name, arguments=args)
                return LLMResponse(text=f"[mock] calling {chosen.name}", tool_calls=[call],
                                   usage=Usage(tin, 8, price(self.model, tin, 8)),
                                   finish_reason="tool_calls")

        context = "\n".join(m.content for m in msgs if m.role == "system")
        observations = [m.content for m in msgs if m.role == "tool"]
        if observations and not re.search(r"\[\d+\]", context):
            text = "[mock] " + " ".join(o[:200] for o in observations[-2:])
        else:
            text = self._synthesise(last_user, context)
        tout = approx_tokens(text)
        return LLMResponse(text=text, usage=Usage(tin, tout, price(self.model, tin, tout)))

    @staticmethod
    def _synthesise(question: str, context: str) -> str:
        """Answer from numbered [n] passages by lexical overlap, or refuse.

        This is not a language model and does not pretend to be one. What it
        does do is exercise the real control flow -- grounded answer, citation
        emission, and refusal when the passages do not cover the question --
        so the offline test suite and the eval harness measure the plumbing
        rather than a stub that always says yes.
        """
        judge_reply = _judge_mock_reply(question, context)
        if judge_reply is not None:
            return judge_reply
        passages = re.findall(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\Z)", context, re.DOTALL)
        if not passages:
            # No numbered passages in the prompt: this is not a grounded-answer
            # call, so just echo. Callers that need specific behaviour should
            # script the response instead.
            return f"[mock] echo: {question.strip()[:300]}"

        stop = {"the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "was", "for",
                "on", "at", "by", "with", "as", "that", "this", "it", "be", "from", "do",
                "does", "how", "what", "when", "who", "why", "which", "can", "i", "my", "me",
                "you", "long", "much", "there", "get", "back", "again", "already"}
        q_terms = {t for t in re.findall(r"[a-z0-9.\-]+", question.lower())
                   if t not in stop and len(t) > 2}
        if not q_terms:
            q_terms = set(re.findall(r"[a-z0-9]+", question.lower()))

        scored = []
        for num, body in passages:
            terms = set(re.findall(r"[a-z0-9.\-]+", body.lower()))
            overlap = len(q_terms & terms) / max(1, len(q_terms))
            scored.append((overlap, int(num), body.strip()))
        scored.sort(reverse=True)

        # Refuse below the coverage floor. Without this the mock answers
        # everything and the negative half of any golden set is untested.
        if scored[0][0] < 0.34:
            return ("NOT_IN_SOURCES The supplied passages do not cover this question; "
                    "a document about this topic would be needed.")

        # Deliberately does NOT attempt to detect "needs clarification" here.
        # Tried a lexical proxy (two passages both clearing the bar, without
        # dominating each other, that don't share much vocabulary) and measured
        # it against the golden set: 9 factoid, 2 multi-hop, 1 aggregation, and
        # 1 unanswerable case fired the marker incorrectly, against 1 correct hit
        # out of 13 real ambiguous cases -- a lexical proxy cannot tell "two
        # sources that should be combined" from "two sources where only one
        # applies" without the semantic judgement a real model has. Rather than
        # ship a heuristic that corrupts citation_recall/faithfulness/
        # answer_correctness on 13 cases to catch 1, clarification_rate reads 0
        # on mock -- same treatment as faithfulness reading near-ceiling on mock:
        # a plumbing check, not a quality estimate. The mechanism itself (prompt,
        # guardrails, API, UI) is tested directly with scripted responses instead
        # of routed through this heuristic -- see tests/test_answer_and_guardrails.py.

        parts = []
        for overlap, num, body in scored[:2]:
            if overlap < 0.34:
                break
            sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", body) if x.strip()]
            best = max(sentences, key=lambda sent: len(
                q_terms & set(re.findall(r"[a-z0-9.\-]+", sent.lower()))), default=body[:200])
            parts.append(f"{best} [{num}]")
        return "[mock] " + " ".join(parts)


def _judge_mock_reply(user_content: str, system_content: str) -> str | None:
    """Recognise core.eval's judge prompts (see core/eval/judge.py) and answer them
    with a deterministic lexical-overlap score instead of falling through to the
    generic echo path, which a judge can never parse as JSON.

    Without this, every judge-based metric (answer_correctness, faithfulness) would
    score a constant 0 on the mock provider -- not because the system under test is
    bad, but because the mock never emits valid JSON. That makes the metric useless
    for CI regression detection. This is lexical, not semantic, like the rest of this
    class: it exercises the judge's parsing and majority-vote control flow, it does
    not estimate real answer quality.
    """
    if '"supported": <fraction of claims supported' in system_content:
        sources, _, answer = user_content.partition("ANSWER:")
        sources = sources.replace("SOURCES:", "", 1)
        s_terms = set(re.findall(r"[a-z0-9]+", sources.lower()))
        a_terms = {t for t in re.findall(r"[a-z0-9]+", answer.lower()) if len(t) > 2}
        overlap = len(a_terms & s_terms) / max(1, len(a_terms))
        return json.dumps({"supported": round(overlap, 2), "claims": []})
    if '"score": <0-4 integer>' in system_content:
        gold_and_candidate = user_content.partition("GOLD ANSWER:")[2]
        gold, _, candidate = gold_and_candidate.partition("CANDIDATE ANSWER:")
        g_terms = {t for t in re.findall(r"[a-z0-9]+", gold.lower()) if len(t) > 2}
        c_terms = {t for t in re.findall(r"[a-z0-9]+", candidate.lower()) if len(t) > 2}
        overlap = len(g_terms & c_terms) / max(1, len(g_terms))
        return json.dumps({"score": round(overlap * 4), "reason": "mock lexical overlap"})
    return None


def _best_tool(question: str, tools: Sequence[ToolSpec]) -> ToolSpec | None:
    """Pick the tool whose name and description overlap the request most."""
    q = set(re.findall(r"[a-z0-9]+", question.lower()))
    if not q:
        return None
    best, best_score = None, 0.0
    for tool in tools:
        if tool.name == "handoff":
            # Routing between agents is a judgement call. A keyword matcher
            # cannot fake it convincingly, and a mock that routes at random
            # produces eval results that mean nothing. Test routing with
            # scripted responses instead (see tests/test_team_memory_api.py).
            continue
        words = set(re.findall(r"[a-z0-9]+", f"{tool.name} {tool.description}".lower()))
        overlap = len(q & words)
        # Name matches count double: "use the calculator" should beat a tool
        # that merely happens to share prose with the question. Prefix matching
        # so "calculator" still finds `calculate`.
        name_words = {w for w in re.split(r"[_.\s]+", tool.name.lower()) if w}
        name_hits = sum(1 for t in q for n in name_words
                        if t == n or (len(t) > 4 and len(n) > 4 and t[:5] == n[:5]))
        score = overlap + 2 * name_hits
        if score > best_score:
            best, best_score = tool, score
    return best if best_score >= 2 else None


def _mock_arguments(tool: ToolSpec, question: str) -> dict[str, Any]:
    """Fabricate plausible arguments from the request text."""
    props = tool.parameters.get("properties", {}) or {}
    required = tool.parameters.get("required", []) or list(props)
    args: dict[str, Any] = {}
    for name in required:
        schema = props.get(name, {})
        if schema.get("enum"):
            q = set(re.findall(r"[a-z0-9]+", question.lower()))
            args[name] = max(schema["enum"],
                             key=lambda v: len(q & set(re.findall(r"[a-z0-9]+", str(v).lower()))))
        elif name in ("expression", "formula"):
            expr = re.search(r"[\d.]+\s*[-+*/%]\s*[\d.()+\-*/ ]+", question)
            args[name] = expr.group(0).strip() if expr else "1+1"
        elif name in ("to", "email", "recipient"):
            found = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", question)
            args[name] = found.group(0) if found else "someone@example.com"
        elif schema.get("type") == "integer":
            found = re.search(r"\b(\d+)\b", question)
            args[name] = int(found.group(1)) if found else 1
        elif schema.get("type") == "number":
            found = re.search(r"\b(\d+(?:\.\d+)?)\b", question)
            args[name] = float(found.group(1)) if found else 1.0
        elif schema.get("type") == "boolean":
            args[name] = False
        elif schema.get("type") == "array":
            args[name] = []
        elif schema.get("type") == "object":
            args[name] = {}
        else:
            args[name] = question[:200]
    return args


class MockEmbeddings:
    """Hashed bag-of-words embedding.

    Not semantic, but it is *lexically* meaningful: documents sharing words get
    higher cosine similarity. That is enough for the retrieval tests to assert
    real ranking behaviour offline instead of asserting on stubs.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.blake2b(token.encode(), digest_size=8).hexdigest(), 16)
                vec[h % self.dim] += 1.0
                vec[(h >> 17) % self.dim] += 0.5
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------


class OpenAILLM:
    def __init__(self, model: str, api_key: str, *, timeout: float = 60.0,
                 base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self._client = httpx.Client(
            base_url=base_url, timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_openai_message(m) for m in messages],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.parameters}}
                for t in tools
            ]
        data = _post(self._client, "/chat/completions", payload)
        choice = data["choices"][0]
        msg = choice["message"]
        calls = [
            ToolCall(id=tc["id"], name=tc["function"]["name"],
                     arguments=json.loads(tc["function"]["arguments"] or "{}"))
            for tc in msg.get("tool_calls") or []
        ]
        u = data.get("usage", {})
        tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        return LLMResponse(
            text=msg.get("content") or "",
            tool_calls=calls,
            usage=Usage(tin, tout, price(self.model, tin, tout)),
            finish_reason=choice.get("finish_reason", "stop"),
        )


class OpenAIEmbeddings:
    def __init__(self, model: str, api_key: str, *, dim: int = 1536, timeout: float = 60.0,
                 base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.dim = dim
        self._client = httpx.Client(base_url=base_url, timeout=timeout,
                                    headers={"Authorization": f"Bearer {api_key}"})

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        batch = 128  # the API rejects very large batches; also caps blast radius on retry
        for i in range(0, len(texts), batch):
            payload = {"model": self.model, "input": list(texts[i:i + batch]), "dimensions": self.dim}
            data = _post(self._client, "/embeddings", payload)
            out.extend(item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"]))
        return out


def _openai_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    d: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in m.tool_calls
        ]
    return d


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


class AnthropicLLM:
    def __init__(self, model: str, api_key: str, *, timeout: float = 60.0,
                 base_url: str = "https://api.anthropic.com/v1"):
        self.model = model
        self._client = httpx.Client(
            base_url=base_url, timeout=timeout,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> LLMResponse:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or 1024,
            "messages": [_anthropic_message(m) for m in messages if m.role != "system"],
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        data = _post(self._client, "/messages", payload)
        text_parts, calls = [], []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(id=block["id"], name=block["name"], arguments=block.get("input") or {}))
        u = data.get("usage", {})
        tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=calls,
            usage=Usage(tin, tout, price(self.model, tin, tout)),
            finish_reason=data.get("stop_reason", "stop"),
        )


def _anthropic_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}]}
    if m.role == "assistant" and m.tool_calls:
        content: list[dict[str, Any]] = []
        if m.content:
            content.append({"type": "text", "text": m.content})
        content += [{"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    for tc in m.tool_calls]
        return {"role": "assistant", "content": content}
    return {"role": m.role, "content": m.content}


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


def _post(client: httpx.Client, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    def once() -> dict[str, Any]:
        try:
            r = client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise RetryableError(f"timeout calling {path}") from exc
        except httpx.HTTPError as exc:
            raise RetryableError(f"transport error calling {path}: {exc}") from exc
        if r.status_code in (408, 409, 429) or r.status_code >= 500:
            raise RetryableError(f"{r.status_code} from {path}: {r.text[:300]}")
        if r.status_code >= 400:
            raise ProviderError(f"{r.status_code} from {path}: {r.text[:500]}")
        return r.json()

    return retry_call(once)


# --------------------------------------------------------------------------
# Budget enforcement + factory
# --------------------------------------------------------------------------


class BudgetedLLM:
    """Wraps any LLMClient and refuses to exceed the per-run budget.

    Several postings describe an agent that looped and burned a month of budget
    overnight. This is the cheapest possible insurance against that, and it is
    a good thing to point at in a proposal.
    """

    def __init__(self, inner: LLMClient, settings: Settings, trace: Trace | None = None):
        self.inner = inner
        self.settings = settings
        self.trace = trace or Trace(enabled=False)

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> LLMResponse:
        c = self.trace.counters
        if c.get("llm_calls", 0) >= self.settings.max_llm_calls_per_run:
            raise ProviderError(
                f"run exceeded max_llm_calls_per_run={self.settings.max_llm_calls_per_run}")
        if c.get("cost_usd", 0.0) >= self.settings.max_cost_usd_per_run:
            raise ProviderError(
                f"run exceeded max_cost_usd_per_run=${self.settings.max_cost_usd_per_run}")
        with self.trace.span("llm.chat", model=getattr(self.inner, "model", "?"),
                             n_messages=len(list(messages))) as span:
            resp = self.inner.chat(
                messages, tools=tools,
                temperature=self.settings.temperature if temperature is None else temperature,
                max_tokens=self.settings.max_output_tokens if max_tokens is None else max_tokens,
            )
            span.attributes.update(tokens_in=resp.usage.tokens_in, tokens_out=resp.usage.tokens_out,
                                   cost_usd=resp.usage.cost_usd, tool_calls=len(resp.tool_calls))
        self.trace.count(llm_calls=1, cost_usd=resp.usage.cost_usd,
                         tokens_in=resp.usage.tokens_in, tokens_out=resp.usage.tokens_out)
        return resp


def build_llm(settings: Settings, trace: Trace | None = None) -> BudgetedLLM:
    if settings.llm_provider == "mock":
        inner: LLMClient = MockLLM(settings.llm_model)
    elif settings.llm_provider == "openai":
        inner = OpenAILLM(settings.llm_model, settings.api_key("openai"),
                          timeout=settings.request_timeout_s)
    elif settings.llm_provider == "anthropic":
        inner = AnthropicLLM(settings.llm_model, settings.api_key("anthropic"),
                             timeout=settings.request_timeout_s)
    elif settings.llm_provider == "ollama":
        inner = OpenAILLM(settings.llm_model, "ollama", timeout=settings.request_timeout_s,
                          base_url=settings.ollama_base_url)
    elif settings.llm_provider == "openrouter":
        inner = OpenAILLM(settings.llm_model, settings.api_key("openrouter"),
                          timeout=settings.request_timeout_s,
                          base_url="https://openrouter.ai/api/v1")
    else:  # pragma: no cover - validate() already rejects this
        raise ProviderError(f"unknown llm_provider {settings.llm_provider!r}")
    return BudgetedLLM(inner, settings, trace)


def build_embeddings(settings: Settings) -> EmbeddingClient:
    if settings.embedding_provider == "mock":
        return MockEmbeddings(settings.embedding_dim)
    if settings.embedding_provider == "ollama":
        return OpenAIEmbeddings(settings.embedding_model, "ollama", dim=settings.embedding_dim,
                                timeout=settings.request_timeout_s, base_url=settings.ollama_base_url)
    if settings.embedding_provider == "openrouter":
        return OpenAIEmbeddings(settings.embedding_model, settings.api_key("openrouter"),
                                dim=settings.embedding_dim, timeout=settings.request_timeout_s,
                                base_url="https://openrouter.ai/api/v1")
    return OpenAIEmbeddings(settings.embedding_model, settings.api_key("openai"),
                            dim=settings.embedding_dim, timeout=settings.request_timeout_s)
