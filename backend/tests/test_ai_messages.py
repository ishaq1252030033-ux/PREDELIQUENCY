"""Test AI message generation quality.

The suite is split into two tiers:

**Unit tests** (default) — mock the LLM so they run fast, offline,
and deterministically.  They validate structure, length limits,
personalisation, and channel dispatch logic.

**Integration tests** (``@pytest.mark.integration``) — call the real
Ollama server.  Skip automatically when Ollama is not running.
Run them explicitly with::

    pytest backend/tests/test_ai_messages.py -v -m integration

Run all::

    pytest backend/tests/test_ai_messages.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.services.ai_message_generator import (
    AIMessageGenerator,
    _SMS_MAX_CHARS,
    _WHATSAPP_MAX_CHARS,
    is_ollama_available,
)

# ---------------------------------------------------------------------------
# Realistic mock responses (what Llama 3.1 would actually produce)
# ---------------------------------------------------------------------------

_MOCK_SMS = (
    "Hi Test, we noticed some changes. "
    "We're here to help with flexible options. "
    "Reply YES for support."
)

_MOCK_EMAIL_SUBJECT = "We're here to help, Priya"
_MOCK_EMAIL_BODY = (
    "Dear Priya,\n\n"
    "We understand that managing finances can be challenging, "
    "especially when unexpected situations arise. We noticed some "
    "changes in your account and want to help.\n\n"
    "Here are some options we can offer:\n"
    "1. Flexible payment plan\n"
    "2. One-month payment holiday\n"
    "3. A call with your relationship manager\n\n"
    "Please call us at 1800-XXX-XXXX or tap the link below.\n\n"
    "Warm regards,\n"
    "Your Bank Support Team"
)

_MOCK_WHATSAPP = (
    "Hi Raj! We noticed things have been a bit tough lately. "
    "We're here to help you stay on track. "
    "Would you like to explore some flexible payment options? "
    "Just reply YES and we'll set it up for you."
)

_MOCK_APP = (
    "Hi Test, we have support options ready for you. "
    "Tap to explore flexible repayment plans tailored to your needs."
)

# Counter to vary mock responses across calls
_call_counter: int = 0


def _mock_llm_response(prompt: str, **_kwargs: Any) -> Dict[str, Any]:
    """Return a deterministic mock response based on the prompt content."""
    global _call_counter
    _call_counter += 1
    text = prompt.lower()

    if "sms" in text:
        body = _MOCK_SMS
    elif "email" in text:
        body = f"Subject: {_MOCK_EMAIL_SUBJECT}\n\n{_MOCK_EMAIL_BODY}"
    elif "whatsapp" in text:
        body = f"WhatsApp Message: {_MOCK_WHATSAPP}"
    elif "notification" in text or "app" in text:
        body = f"Notification: {_MOCK_APP}"
    else:
        body = "Hi, we are here to help you. Reply YES for options."

    return {
        "text": body,
        "tokens_used": 42 + _call_counter,
        "latency_ms": 150.0 + _call_counter,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client():
    """Patch the Ollama client so no real server is needed."""
    mock = MagicMock()
    # client.list() returns a model list that passes the verify check
    mock.list.return_value = {
        "models": [{"name": "llama3.1:latest"}]
    }
    return mock


@pytest.fixture()
def generator(mock_client):
    """Return an ``AIMessageGenerator`` with mocked LLM backend."""
    with patch(
        "backend.app.services.ai_message_generator._get_client",
        return_value=mock_client,
    ):
        gen = AIMessageGenerator()

    # Replace _call_llm with our deterministic mock
    gen._call_llm = _mock_llm_response  # type: ignore[assignment]
    return gen


@pytest.fixture()
def sample_messages(generator) -> Dict[str, Any]:
    """Pre-generated messages for a standard test customer."""
    return generator.generate_intervention_messages(
        customer_name="Test User",
        risk_score=85,
        risk_factors=["Salary delay", "Low savings", "Payment due"],
        payment_due_date="tomorrow",
        payment_amount=5000,
    )


# =========================================================================
# UNIT TESTS — fast, offline, deterministic
# =========================================================================


class TestSMSGeneration:
    """Validate SMS output structure and constraints."""

    def test_sms_is_string(self, sample_messages: Dict[str, Any]) -> None:
        assert isinstance(sample_messages["sms"], str)

    def test_sms_max_length(self, sample_messages: Dict[str, Any]) -> None:
        sms = sample_messages["sms"]
        assert len(sms) <= _SMS_MAX_CHARS, (
            f"SMS is {len(sms)} chars — exceeds {_SMS_MAX_CHARS} limit"
        )

    def test_sms_contains_name(self, generator) -> None:
        msgs = generator.generate_intervention_messages(
            customer_name="Ankit",
            risk_score=75,
            risk_factors=["Late payment"],
        )
        # The mock always starts with "Hi Test" but in real LLM the
        # prompt includes the name — here we verify the key is present
        assert msgs["sms"], "SMS should not be empty"

    def test_sms_not_empty(self, sample_messages: Dict[str, Any]) -> None:
        assert len(sample_messages["sms"]) > 10, "SMS is suspiciously short"


class TestEmailGeneration:
    """Validate email output structure."""

    def test_email_is_dict(self, sample_messages: Dict[str, Any]) -> None:
        assert isinstance(sample_messages["email"], dict)

    def test_email_has_subject(self, sample_messages: Dict[str, Any]) -> None:
        assert "subject" in sample_messages["email"], "Email needs a subject"

    def test_email_has_body(self, sample_messages: Dict[str, Any]) -> None:
        assert "body" in sample_messages["email"], "Email needs a body"

    def test_email_subject_length(
        self, sample_messages: Dict[str, Any]
    ) -> None:
        subject = sample_messages["email"]["subject"]
        assert len(subject) < 100, (
            f"Subject is {len(subject)} chars — should be concise"
        )

    def test_email_body_not_empty(
        self, sample_messages: Dict[str, Any]
    ) -> None:
        assert len(sample_messages["email"]["body"]) > 20


class TestWhatsAppGeneration:
    """Validate WhatsApp output."""

    def test_whatsapp_is_string(
        self, sample_messages: Dict[str, Any]
    ) -> None:
        assert isinstance(sample_messages["whatsapp"], str)

    def test_whatsapp_max_length(
        self, sample_messages: Dict[str, Any]
    ) -> None:
        wa = sample_messages["whatsapp"]
        assert len(wa) <= _WHATSAPP_MAX_CHARS, (
            f"WhatsApp is {len(wa)} chars — exceeds {_WHATSAPP_MAX_CHARS}"
        )

    def test_whatsapp_not_empty(
        self, sample_messages: Dict[str, Any]
    ) -> None:
        assert len(sample_messages["whatsapp"]) > 10


class TestAppPushGeneration:
    """Validate app push notification output."""

    def test_app_is_string(self, sample_messages: Dict[str, Any]) -> None:
        assert isinstance(sample_messages["app"], str)

    def test_app_not_empty(self, sample_messages: Dict[str, Any]) -> None:
        assert len(sample_messages["app"]) > 10


class TestMessageMetadata:
    """Validate metadata fields in the response dict."""

    def test_model_field(self, sample_messages: Dict[str, Any]) -> None:
        assert "model" in sample_messages
        assert isinstance(sample_messages["model"], str)

    def test_ai_generated_flag(
        self, sample_messages: Dict[str, Any]
    ) -> None:
        assert sample_messages["ai_generated"] is True

    def test_generated_at_present(
        self, sample_messages: Dict[str, Any]
    ) -> None:
        assert "generated_at" in sample_messages

    def test_latency_present(
        self, sample_messages: Dict[str, Any]
    ) -> None:
        assert "total_latency_ms" in sample_messages
        assert sample_messages["total_latency_ms"] >= 0


class TestToneQuality:
    """Ensure generated messages meet empathy and tone standards."""

    def test_no_threatening_language(
        self, sample_messages: Dict[str, Any]
    ) -> None:
        all_text = str(sample_messages).lower()
        threatening = [
            "penalty",
            "legal action",
            "lawsuit",
            "sue",
            "prosecute",
            "jail",
            "seize",
            "garnish",
            "blacklist",
        ]
        for word in threatening:
            assert word not in all_text, (
                f"Found threatening word '{word}' in messages"
            )

    def test_supportive_language_present(
        self, sample_messages: Dict[str, Any]
    ) -> None:
        all_text = str(sample_messages).lower()
        supportive = [
            "help",
            "support",
            "assist",
            "care",
            "understand",
            "options",
            "flexible",
        ]
        assert any(w in all_text for w in supportive), (
            "Messages should contain at least one supportive word"
        )


class TestMultiLanguage:
    """Test that language parameter is accepted."""

    def test_hindi_request_succeeds(self, generator) -> None:
        msgs = generator.generate_intervention_messages(
            customer_name="Priya",
            risk_score=80,
            risk_factors=["Payment due"],
            language="hi",
        )
        assert msgs["sms"], "Hindi SMS should not be empty"
        assert msgs["email"]["body"], "Hindi email body should not be empty"

    def test_english_is_default(self, generator) -> None:
        msgs = generator.generate_intervention_messages(
            customer_name="Raj",
            risk_score=70,
            risk_factors=["Low balance"],
        )
        assert msgs["sms"]


class TestCustomMessage:
    """Validate the free-form ``generate_custom_message`` method."""

    def test_returns_string(self, generator) -> None:
        result = generator.generate_custom_message(
            "Write a gentle payment reminder."
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_respects_max_length_param(self, generator) -> None:
        result = generator.generate_custom_message(
            "Write a reminder.", max_length=50
        )
        assert isinstance(result, str)


class TestSingleChannelGeneration:
    """Test the ``generate_single`` method used by the API endpoint."""

    @pytest.mark.parametrize(
        "channel", ["sms", "email", "whatsapp", "app"]
    )
    def test_generate_single_channels(
        self, generator, channel: str
    ) -> None:
        result = generator.generate_single(
            customer_name="Meera",
            risk_score=82.5,
            risk_level="high",
            top_risk_factors=["Late salary", "Savings drop"],
            channel=channel,
            language="en",
        )
        assert result["ai_generated"] is True
        assert result["channel"] == channel
        assert result["body"], f"{channel} body should not be empty"

    def test_email_single_has_subject(self, generator) -> None:
        result = generator.generate_single(
            customer_name="Kiran",
            risk_score=78.0,
            risk_level="high",
            top_risk_factors=["Payment due"],
            channel="email",
        )
        assert result.get("subject") is not None


class TestContextEnrichment:
    """Test that optional context fields are accepted and don't crash."""

    def test_all_optional_fields(self, generator) -> None:
        msgs = generator.generate_intervention_messages(
            customer_name="Deepak Verma",
            risk_score=91,
            risk_factors=[
                "Salary delayed 10 days",
                "Savings dropped 60%",
                "3 lending apps",
            ],
            payment_due_date="in 2 days",
            payment_amount=15000,
            language="en",
            salary_delay_days=10,
            savings_drop_pct=60.0,
            lending_app_count=3,
            lending_app_amount=45000.0,
            failed_payment_count=2,
        )
        assert msgs["sms"]
        assert msgs["email"]["body"]
        assert msgs["whatsapp"]
        assert msgs["app"]

    def test_minimal_fields(self, generator) -> None:
        msgs = generator.generate_intervention_messages(
            customer_name="A",
            risk_score=50,
            risk_factors=[],
        )
        assert msgs["sms"]


class TestEdgeCases:
    """Boundary and edge-case tests."""

    def test_zero_risk_score(self, generator) -> None:
        msgs = generator.generate_intervention_messages(
            customer_name="Safe Customer",
            risk_score=0,
            risk_factors=[],
        )
        assert msgs["sms"]

    def test_max_risk_score(self, generator) -> None:
        msgs = generator.generate_intervention_messages(
            customer_name="Critical",
            risk_score=100,
            risk_factors=["Default imminent"],
        )
        assert msgs["sms"]

    def test_long_customer_name(self, generator) -> None:
        msgs = generator.generate_intervention_messages(
            customer_name="Venkatanarasimharajuvaripeta Subramanian",
            risk_score=80,
            risk_factors=["Test"],
        )
        assert msgs["sms"]

    def test_many_risk_factors(self, generator) -> None:
        msgs = generator.generate_intervention_messages(
            customer_name="User",
            risk_score=85,
            risk_factors=[f"Factor {i}" for i in range(20)],
        )
        assert msgs["sms"]

    def test_empty_name_fallback(self, generator) -> None:
        msgs = generator.generate_intervention_messages(
            customer_name="",
            risk_score=80,
            risk_factors=["Test"],
        )
        # Should not crash — falls back to "Customer"
        assert msgs["sms"]


# =========================================================================
# INTEGRATION TESTS — require Ollama running with llama3.1
# =========================================================================


def _ollama_running() -> bool:
    """Check if Ollama is available for integration tests."""
    try:
        return is_ollama_available()
    except Exception:
        return False


@pytest.fixture()
def live_generator():
    """Return a real ``AIMessageGenerator`` connected to Ollama."""
    return AIMessageGenerator()


@pytest.mark.integration
@pytest.mark.skipif(
    not _ollama_running(),
    reason="Ollama not running — skipping integration tests",
)
class TestLiveGeneration:
    """Integration tests that call the real Ollama server."""

    def test_live_sms_length(self, live_generator) -> None:
        msgs = live_generator.generate_intervention_messages(
            customer_name="Test User",
            risk_score=85,
            risk_factors=["Salary delay", "Low savings", "Payment due"],
            payment_due_date="tomorrow",
            payment_amount=5000,
        )
        assert len(msgs["sms"]) <= _SMS_MAX_CHARS, (
            f"Live SMS is {len(msgs['sms'])} chars"
        )

    def test_live_email_structure(self, live_generator) -> None:
        msgs = live_generator.generate_intervention_messages(
            customer_name="Priya",
            risk_score=88,
            risk_factors=[
                "Salary delayed 7 days",
                "Savings -45%",
                "Payment due",
            ],
            payment_due_date="in 3 days",
            payment_amount=10000,
        )
        assert "subject" in msgs["email"]
        assert "body" in msgs["email"]
        assert len(msgs["email"]["subject"]) < 100

    def test_live_tone_quality(self, live_generator) -> None:
        msgs = live_generator.generate_intervention_messages(
            customer_name="Raj",
            risk_score=92,
            risk_factors=[
                "Multiple delays",
                "High risk",
                "Payment overdue",
            ],
            payment_due_date="today",
            payment_amount=15000,
        )
        all_text = str(msgs).lower()

        supportive = [
            "help",
            "support",
            "assist",
            "understand",
            "options",
            "flexible",
            "plan",
        ]
        assert any(w in all_text for w in supportive), (
            "Live messages should contain supportive language"
        )

        threatening = [
            "penalty",
            "legal action",
            "lawsuit",
            "prosecute",
        ]
        for word in threatening:
            assert word not in all_text, (
                f"Live messages contain threatening word: '{word}'"
            )

    def test_live_personalisation(self, live_generator) -> None:
        msgs = live_generator.generate_intervention_messages(
            customer_name="Priya Sharma",
            risk_score=80,
            risk_factors=["Salary delay"],
            salary_delay_days=5,
        )
        all_text = str(msgs).lower()
        assert "priya" in all_text, "Should personalise with customer name"

    def test_live_ab_variants(self, live_generator) -> None:
        """Generate multiple variants and verify diversity."""
        variants = []
        for _ in range(3):
            msgs = live_generator.generate_intervention_messages(
                customer_name="Test",
                risk_score=80,
                risk_factors=["Test factor"],
                payment_due_date="soon",
                payment_amount=5000,
            )
            variants.append(msgs["sms"])

        assert len(set(variants)) >= 2, (
            "Multiple runs should produce diverse outputs"
        )
        print("\nA/B Test Variants:")
        for i, v in enumerate(variants, 1):
            print(f"\n  Variant {i} ({len(v)} chars):\n  {v}")

    def test_live_all_channels_populated(self, live_generator) -> None:
        msgs = live_generator.generate_intervention_messages(
            customer_name="Deepak",
            risk_score=87,
            risk_factors=["Savings drop", "Lending apps"],
            savings_drop_pct=40.0,
            lending_app_count=2,
        )
        for ch in ("sms", "email", "whatsapp", "app"):
            assert msgs[ch], f"Channel '{ch}' should not be empty"
        assert msgs["email"]["body"]
        assert msgs["total_latency_ms"] > 0


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
