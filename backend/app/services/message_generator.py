"""Personalized intervention message generator with A/B testing and i18n.

Generates context-aware messages for different risk scenarios, channels,
and languages (English and Hindi). Supports A/B variant selection for
measuring message effectiveness.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from backend.app.models.schemas import Channel, RiskLevel
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Language(str, Enum):
    EN = "en"
    HI = "hi"


class Scenario(str, Enum):
    SALARY_DELAY = "salary_delay"
    SAVINGS_DEPLETION = "savings_depletion"
    LENDING_APP_SPIKE = "lending_app_spike"
    PAYMENT_FAILURE = "payment_failure"
    GENERAL_RISK = "general_risk"


class ABVariant(str, Enum):
    A = "A"
    B = "B"


@dataclass
class MessageContext:
    """All data needed to personalize a message."""

    customer_id: str
    customer_name: str
    risk_score: float
    risk_level: RiskLevel
    top_risk_factors: List[str] = field(default_factory=list)
    channel: Channel = Channel.APP
    language: Language = Language.EN
    # Optional fine-grained context
    salary_delay_days: Optional[int] = None
    savings_drop_pct: Optional[float] = None
    lending_app_count: Optional[int] = None
    lending_app_amount: Optional[float] = None
    failed_payment_count: Optional[int] = None
    upcoming_emi_amount: Optional[float] = None
    upcoming_emi_date: Optional[str] = None


@dataclass
class GeneratedMessage:
    """Output of the message generator."""

    customer_id: str
    scenario: Scenario
    channel: Channel
    language: Language
    variant: ABVariant
    subject: Optional[str]  # None for SMS
    body: str
    generated_at: str


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
# Each template is keyed by (scenario, language, variant).
# Placeholders use Python str.format() syntax.
#
# Available placeholders:
#   {name}               Customer first name
#   {risk_score}         Numeric score 0-100
#   {salary_delay_days}  Days salary was late
#   {savings_drop_pct}   Savings depletion %
#   {lending_app_count}  Number of lending apps used
#   {lending_app_amount} Total borrowed via apps (INR)
#   {failed_payments}    Failed payment count
#   {emi_amount}         Upcoming EMI amount (INR)
#   {emi_date}           Upcoming EMI due date
#   {action}             Recommended action text

_TEMPLATES: Dict[tuple, Dict[str, str]] = {
    # ================================================================
    # SALARY DELAY
    # ================================================================
    (Scenario.SALARY_DELAY, Language.EN, ABVariant.A): {
        "subject": "We're here to help, {name}",
        "body": (
            "Hi {name}, we noticed your salary credit was {salary_delay_days} days "
            "later than usual this month. If you're facing cash-flow pressure, we'd "
            "like to help. Tap below to explore flexible repayment options and avoid "
            "late fees on your upcoming dues."
        ),
    },
    (Scenario.SALARY_DELAY, Language.EN, ABVariant.B): {
        "subject": "Important: Your payment schedule",
        "body": (
            "Dear {name}, your recent salary was delayed by {salary_delay_days} days. "
            "To help you stay on track, we can adjust your EMI schedule or offer a "
            "short grace period. Reply YES or visit the app to learn more."
        ),
    },
    (Scenario.SALARY_DELAY, Language.HI, ABVariant.A): {
        "subject": "हम आपकी मदद के लिए हैं, {name}",
        "body": (
            "नमस्ते {name}, हमने देखा कि इस महीने आपकी सैलरी सामान्य से "
            "{salary_delay_days} दिन देर से आई है। अगर आपको कैश-फ्लो में दिक्कत "
            "हो रही है, तो हम मदद कर सकते हैं। लचीले भुगतान विकल्पों के लिए "
            "ऐप पर टैप करें।"
        ),
    },
    (Scenario.SALARY_DELAY, Language.HI, ABVariant.B): {
        "subject": "महत्वपूर्ण: आपका भुगतान शेड्यूल",
        "body": (
            "प्रिय {name}, आपकी हालिया सैलरी {salary_delay_days} दिन देरी से "
            "आई। हम आपकी EMI तारीख बदल सकते हैं या छोटी छूट अवधि दे सकते हैं। "
            "अधिक जानने के लिए YES रिप्लाई करें या ऐप पर जाएं।"
        ),
    },

    # ================================================================
    # SAVINGS DEPLETION
    # ================================================================
    (Scenario.SAVINGS_DEPLETION, Language.EN, ABVariant.A): {
        "subject": "Protect your savings, {name}",
        "body": (
            "Hi {name}, your savings have declined by {savings_drop_pct:.0f}% over the "
            "past few weeks. We understand things can get tight. Here are some steps "
            "to protect your balance and keep your payments on track. {action}"
        ),
    },
    (Scenario.SAVINGS_DEPLETION, Language.EN, ABVariant.B): {
        "subject": "Your savings alert",
        "body": (
            "Dear {name}, we've noticed a {savings_drop_pct:.0f}% drop in your "
            "savings balance. To avoid missing upcoming payments, consider setting "
            "up auto-debit or reviewing your spending plan. We're here to help — "
            "reach out anytime."
        ),
    },
    (Scenario.SAVINGS_DEPLETION, Language.HI, ABVariant.A): {
        "subject": "अपनी बचत सुरक्षित रखें, {name}",
        "body": (
            "नमस्ते {name}, पिछले कुछ हफ्तों में आपकी बचत {savings_drop_pct:.0f}% "
            "कम हुई है। हम समझते हैं कि कभी-कभी स्थिति कठिन हो सकती है। अपने "
            "बैलेंस की सुरक्षा के लिए यहां कुछ कदम हैं। {action}"
        ),
    },
    (Scenario.SAVINGS_DEPLETION, Language.HI, ABVariant.B): {
        "subject": "बचत अलर्ट",
        "body": (
            "प्रिय {name}, आपकी बचत में {savings_drop_pct:.0f}% की गिरावट आई है। "
            "आगामी भुगतान चूकने से बचने के लिए ऑटो-डेबिट सेट करें या अपने खर्चों "
            "की समीक्षा करें। मदद के लिए कभी भी संपर्क करें।"
        ),
    },

    # ================================================================
    # LENDING APP SPIKE
    # ================================================================
    (Scenario.LENDING_APP_SPIKE, Language.EN, ABVariant.A): {
        "subject": "Better borrowing options for you, {name}",
        "body": (
            "Hi {name}, we noticed you've used {lending_app_count} lending apps "
            "recently (totalling ₹{lending_app_amount:,.0f}). High-interest "
            "micro-loans can add up quickly. We can offer you a lower-rate "
            "personal loan to consolidate — tap here to check your eligibility."
        ),
    },
    (Scenario.LENDING_APP_SPIKE, Language.EN, ABVariant.B): {
        "subject": "A smarter way to borrow",
        "body": (
            "Dear {name}, borrowing from multiple lending apps "
            "({lending_app_count} apps, ₹{lending_app_amount:,.0f}) can lead to "
            "a debt spiral. Let us help — we have consolidation and "
            "restructuring options at better rates. Reply HELP or visit the app."
        ),
    },
    (Scenario.LENDING_APP_SPIKE, Language.HI, ABVariant.A): {
        "subject": "आपके लिए बेहतर उधारी विकल्प, {name}",
        "body": (
            "नमस्ते {name}, हमने देखा कि आपने हाल ही में {lending_app_count} "
            "लेंडिंग ऐप्स से कुल ₹{lending_app_amount:,.0f} उधार लिया है। "
            "ज्यादा ब्याज वाले माइक्रो-लोन जल्दी बढ़ सकते हैं। कम ब्याज दर पर "
            "पर्सनल लोन के लिए यहां टैप करें।"
        ),
    },
    (Scenario.LENDING_APP_SPIKE, Language.HI, ABVariant.B): {
        "subject": "उधार लेने का बेहतर तरीका",
        "body": (
            "प्रिय {name}, कई लेंडिंग ऐप्स से उधार ({lending_app_count} ऐप्स, "
            "₹{lending_app_amount:,.0f}) से कर्ज का बोझ बढ़ सकता है। हमारे पास "
            "बेहतर दरों पर समेकन और पुनर्गठन विकल्प हैं। HELP रिप्लाई करें।"
        ),
    },

    # ================================================================
    # PAYMENT FAILURE
    # ================================================================
    (Scenario.PAYMENT_FAILURE, Language.EN, ABVariant.A): {
        "subject": "Let's fix this together, {name}",
        "body": (
            "Hi {name}, we noticed {failed_payments} recent payment attempts didn't "
            "go through. This can affect your credit score. Let's work together — "
            "you can reschedule your payment or set up a partial-payment plan. "
            "Your upcoming EMI of ₹{emi_amount:,.0f} is due on {emi_date}."
        ),
    },
    (Scenario.PAYMENT_FAILURE, Language.EN, ABVariant.B): {
        "subject": "Payment issue — action needed",
        "body": (
            "Dear {name}, {failed_payments} of your recent payments have failed. "
            "To protect your credit standing, please ensure sufficient balance or "
            "contact us to restructure. Next EMI: ₹{emi_amount:,.0f} on {emi_date}."
        ),
    },
    (Scenario.PAYMENT_FAILURE, Language.HI, ABVariant.A): {
        "subject": "आइए इसे मिलकर ठीक करें, {name}",
        "body": (
            "नमस्ते {name}, आपके {failed_payments} हालिया भुगतान प्रयास सफल "
            "नहीं हुए। इससे आपका क्रेडिट स्कोर प्रभावित हो सकता है। भुगतान को "
            "रीशेड्यूल करें या आंशिक-भुगतान योजना बनाएं। आपकी अगली EMI "
            "₹{emi_amount:,.0f} {emi_date} को है।"
        ),
    },
    (Scenario.PAYMENT_FAILURE, Language.HI, ABVariant.B): {
        "subject": "भुगतान समस्या — कार्रवाई आवश्यक",
        "body": (
            "प्रिय {name}, {failed_payments} भुगतान विफल हो गए हैं। अपनी क्रेडिट "
            "स्थिति बचाने के लिए पर्याप्त बैलेंस सुनिश्चित करें या पुनर्गठन के "
            "लिए संपर्क करें। अगली EMI: ₹{emi_amount:,.0f} {emi_date} को।"
        ),
    },

    # ================================================================
    # GENERAL RISK (fallback)
    # ================================================================
    (Scenario.GENERAL_RISK, Language.EN, ABVariant.A): {
        "subject": "Stay on track, {name}",
        "body": (
            "Hi {name}, our system flagged some changes in your account activity "
            "(risk score: {risk_score:.0f}). We want to make sure you don't miss any "
            "payments. Tap here to review your upcoming dues and explore assistance "
            "options. {action}"
        ),
    },
    (Scenario.GENERAL_RISK, Language.EN, ABVariant.B): {
        "subject": "Account update for {name}",
        "body": (
            "Dear {name}, we've noticed changes in your financial pattern. "
            "To help you stay ahead, we recommend reviewing your payment schedule. "
            "Need help? We're just a tap away. {action}"
        ),
    },
    (Scenario.GENERAL_RISK, Language.HI, ABVariant.A): {
        "subject": "सही दिशा में बने रहें, {name}",
        "body": (
            "नमस्ते {name}, हमारे सिस्टम ने आपकी खाता गतिविधि में कुछ बदलाव "
            "देखे हैं (जोखिम स्कोर: {risk_score:.0f})। कोई भुगतान न छूटे, इसके "
            "लिए अपनी आगामी देय राशि की समीक्षा करें। {action}"
        ),
    },
    (Scenario.GENERAL_RISK, Language.HI, ABVariant.B): {
        "subject": "{name} के लिए खाता अपडेट",
        "body": (
            "प्रिय {name}, आपके वित्तीय पैटर्न में बदलाव दिखे हैं। आगे रहने "
            "के लिए अपना भुगतान शेड्यूल जांचें। मदद चाहिए? हम बस एक टैप दूर "
            "हैं। {action}"
        ),
    },
}


# ---------------------------------------------------------------------------
# Channel-specific wrappers (SMS is shorter, email has full subject, etc.)
# ---------------------------------------------------------------------------

_SMS_MAX_LEN = 160  # GSM single-segment limit for reference
_APP_PUSH_MAX_LEN = 256


def _truncate_for_channel(body: str, channel: Channel) -> str:
    """Best-effort truncation for short channels; preserves full text for email."""
    if channel == Channel.SMS and len(body) > _SMS_MAX_LEN:
        return body[: _SMS_MAX_LEN - 3].rstrip() + "..."
    if channel == Channel.APP and len(body) > _APP_PUSH_MAX_LEN:
        return body[: _APP_PUSH_MAX_LEN - 3].rstrip() + "..."
    return body


# ---------------------------------------------------------------------------
# Scenario detection
# ---------------------------------------------------------------------------

_FACTOR_KEYWORDS: Dict[Scenario, List[str]] = {
    Scenario.SALARY_DELAY: ["salary_delay", "days_since_salary", "salary"],
    Scenario.SAVINGS_DEPLETION: [
        "savings_depletion", "balance_trend", "days_to_zero", "avg_balance",
    ],
    Scenario.LENDING_APP_SPIKE: ["lending_app", "lending_app_amount", "lending_app_transactions"],
    Scenario.PAYMENT_FAILURE: ["failed_payment", "utility_payment_delay", "payment_timing"],
}


def detect_scenario(risk_factors: List[str]) -> Scenario:
    """Detect the primary risk scenario from top risk factor strings."""
    factors_lower = " ".join(risk_factors).lower()
    scores: Dict[Scenario, int] = {s: 0 for s in Scenario if s != Scenario.GENERAL_RISK}
    for scenario, keywords in _FACTOR_KEYWORDS.items():
        for kw in keywords:
            if kw in factors_lower:
                scores[scenario] += 1

    best = max(scores, key=lambda s: scores[s])
    if scores[best] > 0:
        return best
    return Scenario.GENERAL_RISK


# ---------------------------------------------------------------------------
# A/B variant assignment
# ---------------------------------------------------------------------------

def assign_variant(customer_id: str) -> ABVariant:
    """Deterministic A/B split based on customer_id hash (50/50)."""
    h = hashlib.sha256(customer_id.encode("utf-8")).hexdigest()
    return ABVariant.A if int(h, 16) % 2 == 0 else ABVariant.B


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MessageGenerator:
    """Generate personalized intervention messages."""

    def generate(
        self,
        ctx: MessageContext,
        variant_override: Optional[ABVariant] = None,
        scenario_override: Optional[Scenario] = None,
    ) -> GeneratedMessage:
        """
        Build a personalized message for the given context.

        Args:
            ctx: Customer and risk context.
            variant_override: Force A or B (ignores hash-based assignment).
            scenario_override: Force a specific scenario instead of auto-detecting.

        Returns:
            GeneratedMessage with fully rendered body and metadata.
        """
        scenario = scenario_override or detect_scenario(ctx.top_risk_factors)
        variant = variant_override or assign_variant(ctx.customer_id)

        key = (scenario, ctx.language, variant)
        template = _TEMPLATES.get(key)
        if template is None:
            # Fallback chain: general_risk -> EN -> variant A
            for fallback_key in [
                (Scenario.GENERAL_RISK, ctx.language, variant),
                (Scenario.GENERAL_RISK, Language.EN, ABVariant.A),
            ]:
                template = _TEMPLATES.get(fallback_key)
                if template is not None:
                    break
        if template is None:
            raise ValueError(f"No template found for {key}")

        # Build substitution dict
        first_name = ctx.customer_name.split()[0] if ctx.customer_name else "Customer"
        action = _action_for_level(ctx.risk_level, ctx.language)

        subs = {
            "name": first_name,
            "risk_score": ctx.risk_score,
            "salary_delay_days": ctx.salary_delay_days or "several",
            "savings_drop_pct": ctx.savings_drop_pct or 0.0,
            "lending_app_count": ctx.lending_app_count or 0,
            "lending_app_amount": ctx.lending_app_amount or 0.0,
            "failed_payments": ctx.failed_payment_count or 0,
            "emi_amount": ctx.upcoming_emi_amount or 0.0,
            "emi_date": ctx.upcoming_emi_date or "your next due date",
            "action": action,
        }

        body = _safe_format(template["body"], subs)
        body = _truncate_for_channel(body, ctx.channel)

        subject_raw = template.get("subject")
        subject = _safe_format(subject_raw, subs) if subject_raw else None
        # SMS has no subject
        if ctx.channel == Channel.SMS:
            subject = None

        return GeneratedMessage(
            customer_id=ctx.customer_id,
            scenario=scenario,
            channel=ctx.channel,
            language=ctx.language,
            variant=variant,
            subject=subject,
            body=body,
            generated_at=datetime.utcnow().isoformat(),
        )

    def generate_all_variants(
        self,
        ctx: MessageContext,
        scenario_override: Optional[Scenario] = None,
    ) -> Dict[str, GeneratedMessage]:
        """Generate both A and B variants for comparison / preview."""
        return {
            "A": self.generate(ctx, variant_override=ABVariant.A, scenario_override=scenario_override),
            "B": self.generate(ctx, variant_override=ABVariant.B, scenario_override=scenario_override),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_for_level(level: RiskLevel, lang: Language) -> str:
    """Return a recommended action string based on risk level and language."""
    if lang == Language.HI:
        if level == RiskLevel.LOW:
            return "कोई तत्काल कार्रवाई आवश्यक नहीं।"
        if level == RiskLevel.MEDIUM:
            return "अपनी आगामी देय राशि की समीक्षा करें।"
        if level == RiskLevel.HIGH:
            return "लचीले भुगतान विकल्पों के लिए ऐप पर टैप करें।"
        return "हमारी टीम से बात करने के लिए अभी कॉल करें।"
    # English
    if level == RiskLevel.LOW:
        return "No immediate action required."
    if level == RiskLevel.MEDIUM:
        return "Review your upcoming dues in the app."
    if level == RiskLevel.HIGH:
        return "Tap here to explore flexible repayment options."
    return "Call us now to discuss a recovery plan."


def _safe_format(template: str, subs: dict) -> str:
    """Format a template string, gracefully ignoring missing keys."""
    try:
        return template.format(**subs)
    except (KeyError, ValueError, IndexError) as exc:
        logger.warning("Template formatting issue: %s", exc)
        # Partial fallback: replace what we can
        for k, v in subs.items():
            template = template.replace(f"{{{k}}}", str(v))
        return template
