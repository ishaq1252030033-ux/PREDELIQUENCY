"""AI Message Generator using Open-Source Llama 3.1.

NO API costs, 100% private, GDPR compliant.

Setup (one-time, ~5 minutes)::

    # 1. Install Ollama — https://ollama.com
    # 2. Pull the model (~4 GB download)
    ollama pull llama3.1:8b-instruct-q4_0

    # 3. Start the server (or use the Ollama desktop app)
    ollama serve

Everything runs **locally** — no API keys, no cloud, no internet needed
after the initial model download.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Ollama configuration (overridable via environment variables)
# ---------------------------------------------------------------------------

OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT: int = int(os.environ.get("OLLAMA_TIMEOUT", "60"))

# Channel-specific limits
_SMS_MAX_CHARS: int = 160
_WHATSAPP_MAX_CHARS: int = 500
_EMAIL_MAX_TOKENS: int = 300
_APP_MAX_TOKENS: int = 150


def _get_client() -> Any:
    """Lazily import and return an Ollama ``Client``.

    Returns:
        An ``ollama.Client`` instance pointed at ``OLLAMA_HOST``.

    Raises:
        ImportError: If the ``ollama`` package is not installed.
    """
    from ollama import Client  # type: ignore[import-untyped]

    return Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)


# ---------------------------------------------------------------------------
# Health / availability helpers
# ---------------------------------------------------------------------------


def is_ollama_available() -> bool:
    """Return ``True`` if the Ollama server is reachable and the configured model is pulled.

    Returns:
        Boolean availability flag.
    """
    try:
        client = _get_client()
        models = client.list()
        model_names: list[str] = []
        if hasattr(models, "models"):
            model_names = [m.model for m in models.models]
        elif isinstance(models, dict):
            model_names = [
                m.get("name", "") for m in models.get("models", [])
            ]
        return any(OLLAMA_MODEL in n for n in model_names)
    except Exception as exc:
        logger.debug("Ollama not available: %s", exc)
        return False


def list_available_models() -> List[str]:
    """Return a list of model names currently available in Ollama.

    Returns:
        List of model name strings, or empty list on failure.
    """
    try:
        client = _get_client()
        models = client.list()
        if hasattr(models, "models"):
            return [m.model for m in models.models]
        if isinstance(models, dict):
            return [
                m.get("name", "") for m in models.get("models", [])
            ]
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# System prompt — tuned for Indian banking intervention messages
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = """\
You are a banking intervention message writer for a Pre-Delinquency \
Intervention Engine at an Indian bank. Your job is to write SHORT, \
empathetic, personalised messages that help customers avoid defaulting \
on their payments.

Rules:
- Be warm, supportive, and non-threatening
- NEVER shame or blame the customer
- Include specific numbers when provided (amounts in INR, dates, days)
- If language is Hindi, write ENTIRELY in Hindi (Devanagari script)
- Always include a clear call-to-action
- Reference the specific risk scenario (salary delay, savings drop, etc.)
- Mention the customer's name naturally
- Sound human, not robotic — like a caring relationship manager
- Offer concrete solutions (flexible plans, grace periods, helpline)
"""


# ---------------------------------------------------------------------------
# AIMessageGenerator class
# ---------------------------------------------------------------------------


class AIMessageGenerator:
    """Generate personalised intervention messages using Llama 3.1 (FREE).

    Supports multi-channel output (SMS, email, WhatsApp, app push) and
    multi-language (English, Hindi). All inference runs locally via
    Ollama — no API costs, fully GDPR-compliant.

    Attributes:
        model: The Ollama model tag to use.
        client: The Ollama client instance.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        """Initialise the generator and verify the model is available.

        Args:
            model: Override the default ``OLLAMA_MODEL``. Pass any
                model tag that you've pulled into Ollama (e.g.
                ``"llama3.1:8b-instruct-q4_0"``, ``"mistral"``).
        """
        self.model: str = model or OLLAMA_MODEL
        self.client = _get_client()
        self._verify_model()

    def _verify_model(self) -> None:
        """Check if the configured model is available in Ollama."""
        try:
            models = self.client.list()
            model_names: list[str] = []
            if hasattr(models, "models"):
                model_names = [m.model for m in models.models]
            elif isinstance(models, dict):
                model_names = [
                    m.get("name", "") for m in models.get("models", [])
                ]

            if any(self.model in n for n in model_names):
                logger.info("AI model ready: %s", self.model)
            else:
                logger.warning(
                    "Model '%s' not found. Available: %s. "
                    "Run: ollama pull %s",
                    self.model,
                    model_names,
                    self.model,
                )
        except Exception as exc:
            logger.error("Cannot connect to Ollama: %s", exc)

    # ------------------------------------------------------------------ #
    # Core generation
    # ------------------------------------------------------------------ #

    def _call_llm(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Send a prompt to the local LLM and return the raw response.

        Args:
            prompt: The user-turn prompt text.
            temperature: Sampling temperature (0-1).
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum tokens to generate.

        Returns:
            Dict with ``text`` (generated string), ``tokens_used``,
            and ``latency_ms``.

        Raises:
            RuntimeError: If Ollama is unreachable or the call fails.
        """
        t0 = time.perf_counter()
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
            )
        except Exception as exc:
            logger.exception("Ollama call failed: %s", exc)
            raise RuntimeError(
                f"Ollama generation failed: {exc}. "
                "Is Ollama running? Try: ollama serve"
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract text
        if hasattr(response, "message"):
            text = response.message.content or ""
        elif isinstance(response, dict):
            text = response.get("message", {}).get("content", "")
        else:
            text = str(response)

        # Token usage
        tokens = 0
        if hasattr(response, "eval_count"):
            tokens = response.eval_count
        elif isinstance(response, dict):
            tokens = response.get("eval_count", 0)

        return {
            "text": text.strip(),
            "tokens_used": tokens,
            "latency_ms": round(latency_ms, 1),
        }

    # ------------------------------------------------------------------ #
    # Multi-channel generation
    # ------------------------------------------------------------------ #

    def generate_intervention_messages(
        self,
        customer_name: str,
        risk_score: float,
        risk_factors: List[str],
        payment_due_date: str = "your next due date",
        payment_amount: float = 0.0,
        language: str = "en",
        salary_delay_days: Optional[int] = None,
        savings_drop_pct: Optional[float] = None,
        lending_app_count: Optional[int] = None,
        lending_app_amount: Optional[float] = None,
        failed_payment_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate intervention messages for **all channels** at once.

        Args:
            customer_name: Customer's first name or full name.
            risk_score: Risk score 0-100.
            risk_factors: List of top risk factors.
            payment_due_date: When payment is due (e.g. "in 3 days").
            payment_amount: Amount due in INR.
            language: ``"en"`` or ``"hi"``.
            salary_delay_days: Days salary was late.
            savings_drop_pct: Savings depletion percentage.
            lending_app_count: Number of lending apps used.
            lending_app_amount: Total borrowed via apps (INR).
            failed_payment_count: Failed payment count.

        Returns:
            Dict with ``sms``, ``email`` (dict with subject + body),
            ``whatsapp``, ``app``, plus ``model``, ``ai_generated``,
            ``generated_at``, and ``total_latency_ms``.
        """
        first_name = customer_name.split()[0] if customer_name else "Customer"
        lang_label = "Hindi (Devanagari)" if language == "hi" else "English"

        # Build shared context block
        context_lines = [
            f"Customer: {first_name}",
            f"Risk Score: {risk_score:.0f}/100",
            f"Issues Detected: {', '.join(risk_factors) if risk_factors else 'General risk'}",
        ]
        if payment_amount > 0:
            context_lines.append(
                f"Payment Due: {payment_due_date} | Amount: \u20b9{payment_amount:,.0f}"
            )
        if salary_delay_days:
            context_lines.append(f"Salary Delay: {salary_delay_days} days late")
        if savings_drop_pct:
            context_lines.append(f"Savings Drop: {savings_drop_pct:.0f}%")
        if lending_app_count:
            amt = (
                f", total \u20b9{lending_app_amount:,.0f}"
                if lending_app_amount
                else ""
            )
            context_lines.append(
                f"Lending Apps: {lending_app_count} apps{amt}"
            )
        if failed_payment_count:
            context_lines.append(
                f"Failed Payments: {failed_payment_count}"
            )

        context = "\n".join(context_lines)
        total_t0 = time.perf_counter()

        # Generate each channel
        sms = self._generate_sms(context, first_name, lang_label)
        email = self._generate_email(context, first_name, lang_label)
        whatsapp = self._generate_whatsapp(context, first_name, lang_label)
        app_msg = self._generate_app(context, first_name, lang_label)

        total_ms = (time.perf_counter() - total_t0) * 1000

        return {
            "sms": sms,
            "email": email,
            "whatsapp": whatsapp,
            "app": app_msg,
            "model": self.model,
            "ai_generated": True,
            "generated_at": datetime.now().isoformat(),
            "total_latency_ms": round(total_ms, 1),
        }

    # ------------------------------------------------------------------ #
    # Channel-specific generators
    # ------------------------------------------------------------------ #

    def _generate_sms(
        self, context: str, name: str, lang: str
    ) -> str:
        """Generate an SMS message (max 160 characters).

        Args:
            context: Shared customer context block.
            name: Customer first name.
            lang: Language label for the prompt.

        Returns:
            SMS text string.
        """
        prompt = (
            f"{context}\n\n"
            f"Generate a caring SMS intervention message in {lang}:\n"
            f"- MAX 160 characters (CRITICAL — count carefully)\n"
            f'- Start with "Hi {name}"\n'
            f"- Mention the specific issue detected\n"
            f"- Offer help, not threats\n"
            f'- End with "Reply YES for options" or similar\n'
            f"- Warm, supportive tone\n\n"
            f"SMS:"
        )
        result = self._call_llm(
            prompt, temperature=0.7, max_tokens=60
        )
        sms = result["text"]
        if sms.startswith("SMS:"):
            sms = sms[4:].strip()
        # Hard limit
        if len(sms) > _SMS_MAX_CHARS:
            sms = sms[: _SMS_MAX_CHARS - 3].rstrip() + "..."
        return sms

    def _generate_email(
        self, context: str, name: str, lang: str
    ) -> Dict[str, str]:
        """Generate an email with subject and body.

        Args:
            context: Shared customer context block.
            name: Customer first name.
            lang: Language label for the prompt.

        Returns:
            Dict with ``subject`` and ``body`` keys.
        """
        prompt = (
            f"{context}\n\n"
            f"Generate a professional but caring email in {lang}:\n\n"
            f"SUBJECT LINE (max 50 chars):\n"
            f"- Mention support/help available\n"
            f"- Urgent but not threatening\n\n"
            f"EMAIL BODY:\n"
            f"- Personal greeting to {name}\n"
            f"- Acknowledge specific issues (salary delay, savings drop, etc.)\n"
            f"- Express empathy\n"
            f"- Offer 3 specific solutions:\n"
            f"  1. Flexible payment plan\n"
            f"  2. Payment holiday (1 month)\n"
            f"  3. Speak to relationship manager\n"
            f"- Call to action (click link or call helpline)\n"
            f"- Sign off warmly\n\n"
            f"Subject:\nBody:"
        )
        result = self._call_llm(
            prompt, temperature=0.6, max_tokens=_EMAIL_MAX_TOKENS
        )
        raw = result["text"]

        # Parse subject and body
        subject = ""
        body = raw
        for prefix in (
            "Subject:",
            "**Subject:**",
            "Subject Line:",
        ):
            if raw.startswith(prefix):
                parts = raw.split("\n", 1)
                subject = parts[0].replace(prefix, "").strip().strip("*")
                body = parts[1].strip() if len(parts) > 1 else raw
                break

        # Remove "Body:" prefix if present
        if body.startswith("Body:"):
            body = body[5:].strip()

        return {"subject": subject, "body": body}

    def _generate_whatsapp(
        self, context: str, name: str, lang: str
    ) -> str:
        """Generate a WhatsApp message (conversational).

        Args:
            context: Shared customer context block.
            name: Customer first name.
            lang: Language label for the prompt.

        Returns:
            WhatsApp message text.
        """
        prompt = (
            f"{context}\n\n"
            f"Generate a friendly WhatsApp message in {lang}:\n"
            f"- Very conversational (like talking to a friend)\n"
            f"- Use emojis sparingly (1-2 max)\n"
            f"- Short paragraphs\n"
            f"- Mention the specific issue\n"
            f"- Offer help and a specific next step\n"
            f"- End with easy response option\n"
            f"- Max 3-4 short paragraphs\n\n"
            f"WhatsApp Message:"
        )
        result = self._call_llm(
            prompt, temperature=0.75, max_tokens=_APP_MAX_TOKENS
        )
        msg = result["text"]
        if msg.startswith("WhatsApp Message:"):
            msg = msg[17:].strip()
        if len(msg) > _WHATSAPP_MAX_CHARS:
            msg = msg[: _WHATSAPP_MAX_CHARS - 3].rstrip() + "..."
        return msg

    def _generate_app(
        self, context: str, name: str, lang: str
    ) -> str:
        """Generate an app push notification (2-3 sentences).

        Args:
            context: Shared customer context block.
            name: Customer first name.
            lang: Language label for the prompt.

        Returns:
            App notification text.
        """
        prompt = (
            f"{context}\n\n"
            f"Generate an app push notification in {lang}:\n"
            f"- 2-3 sentences maximum\n"
            f"- Caring and helpful tone\n"
            f"- Reference the specific risk detected\n"
            f'- End with a CTA like "Tap to explore options"\n\n'
            f"Notification:"
        )
        result = self._call_llm(
            prompt, temperature=0.7, max_tokens=_APP_MAX_TOKENS
        )
        msg = result["text"]
        if msg.startswith("Notification:"):
            msg = msg[13:].strip()
        return msg

    # ------------------------------------------------------------------ #
    # Single-channel generation (used by the API endpoint)
    # ------------------------------------------------------------------ #

    def generate_single(
        self,
        customer_name: str,
        risk_score: float,
        risk_level: str,
        top_risk_factors: List[str],
        channel: str = "app",
        language: str = "en",
        salary_delay_days: Optional[int] = None,
        savings_drop_pct: Optional[float] = None,
        lending_app_count: Optional[int] = None,
        lending_app_amount: Optional[float] = None,
        failed_payment_count: Optional[int] = None,
        upcoming_emi_amount: Optional[float] = None,
        upcoming_emi_date: Optional[str] = None,
        scenario: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate one message for a specific channel.

        This is the method called by the ``/ai/generate-message`` API
        endpoint.  It builds the context and dispatches to the
        appropriate channel generator.

        Args:
            customer_name: Customer display name.
            risk_score: 0-100 risk score.
            risk_level: low / medium / high / critical.
            top_risk_factors: SHAP-based factor list.
            channel: sms / email / app / whatsapp.
            language: en / hi.
            salary_delay_days: Optional salary delay context.
            savings_drop_pct: Optional savings drop context.
            lending_app_count: Optional lending app count.
            lending_app_amount: Optional lending app total.
            failed_payment_count: Optional failed payment count.
            upcoming_emi_amount: Optional EMI amount.
            upcoming_emi_date: Optional EMI date.
            scenario: Detected risk scenario.

        Returns:
            Dict with ``subject``, ``body``, ``model``,
            ``ai_generated``, ``tokens_used``, and ``latency_ms``.
        """
        first_name = (
            customer_name.split()[0] if customer_name else "Customer"
        )
        lang_label = (
            "Hindi (Devanagari)" if language == "hi" else "English"
        )

        # Build context
        context_lines = [
            f"Customer: {first_name}",
            f"Risk Score: {risk_score:.1f}/100 ({risk_level})",
            f"Top Risk Factors: {', '.join(top_risk_factors) if top_risk_factors else 'General risk'}",
        ]
        if scenario:
            context_lines.append(f"Detected Scenario: {scenario}")
        if salary_delay_days:
            context_lines.append(
                f"Salary Delay: {salary_delay_days} days late"
            )
        if savings_drop_pct:
            context_lines.append(f"Savings Drop: {savings_drop_pct:.0f}%")
        if lending_app_count:
            amt = (
                f", total \u20b9{lending_app_amount:,.0f}"
                if lending_app_amount
                else ""
            )
            context_lines.append(
                f"Lending Apps: {lending_app_count} apps{amt}"
            )
        if failed_payment_count:
            context_lines.append(
                f"Failed Payments: {failed_payment_count}"
            )
        if upcoming_emi_amount:
            emi_extra = (
                f" due {upcoming_emi_date}" if upcoming_emi_date else ""
            )
            context_lines.append(
                f"Upcoming EMI: \u20b9{upcoming_emi_amount:,.0f}{emi_extra}"
            )

        context = "\n".join(context_lines)

        # Dispatch to channel
        subject: Optional[str] = None
        if channel == "sms":
            body = self._generate_sms(context, first_name, lang_label)
        elif channel == "email":
            email_result = self._generate_email(
                context, first_name, lang_label
            )
            subject = email_result["subject"]
            body = email_result["body"]
        elif channel == "whatsapp":
            body = self._generate_whatsapp(
                context, first_name, lang_label
            )
        else:
            body = self._generate_app(context, first_name, lang_label)

        return {
            "subject": subject,
            "body": body,
            "model": self.model,
            "ai_generated": True,
            "tokens_used": 0,
            "channel": channel,
            "language": language,
        }

    # ------------------------------------------------------------------ #
    # Custom prompt generation
    # ------------------------------------------------------------------ #

    def generate_custom_message(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """Generate a custom message for any scenario.

        Useful for A/B testing different tones, experimenting with
        new intervention strategies, or generating one-off messages.

        Args:
            prompt: Free-form prompt describing what to generate.
            max_length: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text string.
        """
        result = self._call_llm(
            prompt,
            temperature=temperature,
            max_tokens=max_length,
        )
        return result["text"]


# ---------------------------------------------------------------------------
# Module-level convenience functions (used by routes.py)
# ---------------------------------------------------------------------------

_generator_instance: Optional[AIMessageGenerator] = None


def _get_generator() -> AIMessageGenerator:
    """Return a lazily-initialised singleton generator.

    Returns:
        The shared ``AIMessageGenerator`` instance.
    """
    global _generator_instance  # noqa: PLW0603
    if _generator_instance is None:
        _generator_instance = AIMessageGenerator()
    return _generator_instance


def generate_ai_message(
    customer_name: str,
    risk_score: float,
    risk_level: str,
    top_risk_factors: List[str],
    channel: str = "app",
    language: str = "en",
    salary_delay_days: Optional[int] = None,
    savings_drop_pct: Optional[float] = None,
    lending_app_count: Optional[int] = None,
    lending_app_amount: Optional[float] = None,
    failed_payment_count: Optional[int] = None,
    upcoming_emi_amount: Optional[float] = None,
    upcoming_emi_date: Optional[str] = None,
    scenario: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Module-level convenience function for single-channel AI generation.

    Called by the ``POST /api/v1/ai/generate-message`` endpoint.

    Args:
        customer_name: Customer display name.
        risk_score: 0-100 risk score.
        risk_level: Risk bucket string.
        top_risk_factors: SHAP-based factor list.
        channel: sms / email / app / whatsapp.
        language: en / hi.
        salary_delay_days: Optional salary delay.
        savings_drop_pct: Optional savings drop.
        lending_app_count: Optional lending app count.
        lending_app_amount: Optional lending app total.
        failed_payment_count: Optional failed payments.
        upcoming_emi_amount: Optional EMI amount.
        upcoming_emi_date: Optional EMI date.
        scenario: Detected scenario.
        model: Override model tag.

    Returns:
        Dict with ``subject``, ``body``, ``model``, ``ai_generated``,
        ``tokens_used``, ``channel``, ``language``.
    """
    gen = _get_generator()
    if model and model != gen.model:
        gen = AIMessageGenerator(model=model)

    return gen.generate_single(
        customer_name=customer_name,
        risk_score=risk_score,
        risk_level=risk_level,
        top_risk_factors=top_risk_factors,
        channel=channel,
        language=language,
        salary_delay_days=salary_delay_days,
        savings_drop_pct=savings_drop_pct,
        lending_app_count=lending_app_count,
        lending_app_amount=lending_app_amount,
        failed_payment_count=failed_payment_count,
        upcoming_emi_amount=upcoming_emi_amount,
        upcoming_emi_date=upcoming_emi_date,
        scenario=scenario,
    )


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("AI Message Generator — Llama 3.1 (FREE, LOCAL, PRIVATE)")
    print("=" * 60)

    generator = AIMessageGenerator()

    # Test case: Priya Sharma scenario
    print("\nGenerating messages for test customer...\n")

    messages = generator.generate_intervention_messages(
        customer_name="Priya Sharma",
        risk_score=88,
        risk_factors=[
            "Salary delayed 7 days",
            "Savings dropped 45%",
            "Credit card payment due soon",
        ],
        payment_due_date="in 3 days",
        payment_amount=5000,
        salary_delay_days=7,
        savings_drop_pct=45,
    )

    print("\U0001f4f1 SMS:")
    print(messages["sms"])
    print(f"   ({len(messages['sms'])} characters)")

    print("\n\U0001f4e7 EMAIL:")
    print(f"   Subject: {messages['email']['subject']}")
    print(f"   Body:\n{messages['email']['body']}")

    print("\n\U0001f4ac WhatsApp:")
    print(messages["whatsapp"])

    print("\n\U0001f4f2 App Push:")
    print(messages["app"])

    print(f"\n\u23f1 Total latency: {messages['total_latency_ms']:.0f} ms")
    print(f"\U0001f916 Model: {messages['model']}")
    print(f"\U0001f7e2 AI Generated: {messages['ai_generated']}")
