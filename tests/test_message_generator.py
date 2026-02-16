"""Tests for the message generator service."""

from __future__ import annotations

import pytest

from backend.app.models.schemas import Channel, RiskLevel
from backend.app.services.message_generator import (
    ABVariant,
    GeneratedMessage,
    Language,
    MessageContext,
    MessageGenerator,
    Scenario,
    assign_variant,
    detect_scenario,
)


# ---------------------------------------------------------------------------
# detect_scenario
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "factors, expected",
    [
        (["salary_delay (↑, |SHAP|=0.12)"], Scenario.SALARY_DELAY),
        (["days_since_salary (↑)"], Scenario.SALARY_DELAY),
        (["lending_app_amount (↑)"], Scenario.LENDING_APP_SPIKE),
        (["lending_app_transactions (↑)"], Scenario.LENDING_APP_SPIKE),
        (["savings_depletion_rate (↑)", "balance_trend (↓)"], Scenario.SAVINGS_DEPLETION),
        (["failed_payments_count (↑)"], Scenario.PAYMENT_FAILURE),
        (["utility_payment_delay (↑)"], Scenario.PAYMENT_FAILURE),
        (["nothing_relevant"], Scenario.GENERAL_RISK),
        ([], Scenario.GENERAL_RISK),
    ],
)
def test_detect_scenario(factors: list[str], expected: Scenario) -> None:
    assert detect_scenario(factors) == expected


# ---------------------------------------------------------------------------
# assign_variant
# ---------------------------------------------------------------------------

def test_assign_variant_deterministic() -> None:
    """Same customer_id always yields the same variant."""
    v1 = assign_variant("C000001")
    v2 = assign_variant("C000001")
    assert v1 == v2
    assert v1 in (ABVariant.A, ABVariant.B)


def test_assign_variant_coverage() -> None:
    """Over many IDs we should see both A and B."""
    variants = {assign_variant(f"C{i:06d}") for i in range(200)}
    assert ABVariant.A in variants
    assert ABVariant.B in variants


# ---------------------------------------------------------------------------
# MessageGenerator.generate
# ---------------------------------------------------------------------------

@pytest.fixture()
def gen() -> MessageGenerator:
    return MessageGenerator()


@pytest.fixture()
def salary_ctx() -> MessageContext:
    return MessageContext(
        customer_id="DEMO01",
        customer_name="Priya Sharma",
        risk_score=88.0,
        risk_level=RiskLevel.CRITICAL,
        top_risk_factors=["salary_delay (↑, |SHAP|=0.12)"],
        channel=Channel.APP,
        language=Language.EN,
        salary_delay_days=7,
    )


def test_generate_returns_generated_message(gen: MessageGenerator, salary_ctx: MessageContext) -> None:
    msg = gen.generate(salary_ctx)
    assert isinstance(msg, GeneratedMessage)
    assert msg.customer_id == "DEMO01"
    assert msg.scenario == Scenario.SALARY_DELAY
    assert msg.language == Language.EN


def test_personalization_substitution(gen: MessageGenerator, salary_ctx: MessageContext) -> None:
    msg = gen.generate(salary_ctx)
    assert "Priya" in msg.body
    assert "7 days" in msg.body


def test_hindi_message(gen: MessageGenerator) -> None:
    ctx = MessageContext(
        customer_id="DEMO02",
        customer_name="Raj Kumar",
        risk_score=92.0,
        risk_level=RiskLevel.CRITICAL,
        top_risk_factors=["lending_app_amount (↑)"],
        channel=Channel.EMAIL,
        language=Language.HI,
        lending_app_count=3,
        lending_app_amount=45000,
    )
    msg = gen.generate(ctx)
    assert msg.language == Language.HI
    assert msg.scenario == Scenario.LENDING_APP_SPIKE
    assert msg.subject is not None


def test_sms_has_no_subject(gen: MessageGenerator) -> None:
    ctx = MessageContext(
        customer_id="DEMO03",
        customer_name="Anita Verma",
        risk_score=76.0,
        risk_level=RiskLevel.HIGH,
        top_risk_factors=["savings_depletion_rate (↑)"],
        channel=Channel.SMS,
        language=Language.EN,
        savings_drop_pct=45.0,
    )
    msg = gen.generate(ctx)
    assert msg.subject is None


def test_sms_body_truncation(gen: MessageGenerator) -> None:
    """SMS body should not exceed 160 chars."""
    ctx = MessageContext(
        customer_id="DEMO03",
        customer_name="Anita Verma",
        risk_score=76.0,
        risk_level=RiskLevel.HIGH,
        top_risk_factors=["savings_depletion_rate (↑)"],
        channel=Channel.SMS,
        language=Language.EN,
        savings_drop_pct=45.0,
    )
    msg = gen.generate(ctx)
    assert len(msg.body) <= 160


def test_variant_override(gen: MessageGenerator, salary_ctx: MessageContext) -> None:
    msg_a = gen.generate(salary_ctx, variant_override=ABVariant.A)
    msg_b = gen.generate(salary_ctx, variant_override=ABVariant.B)
    assert msg_a.variant == ABVariant.A
    assert msg_b.variant == ABVariant.B
    assert msg_a.body != msg_b.body


def test_scenario_override(gen: MessageGenerator, salary_ctx: MessageContext) -> None:
    msg = gen.generate(salary_ctx, scenario_override=Scenario.PAYMENT_FAILURE)
    assert msg.scenario == Scenario.PAYMENT_FAILURE


def test_generate_all_variants(gen: MessageGenerator, salary_ctx: MessageContext) -> None:
    both = gen.generate_all_variants(salary_ctx)
    assert "A" in both and "B" in both
    assert both["A"].variant == ABVariant.A
    assert both["B"].variant == ABVariant.B


def test_payment_failure_amounts(gen: MessageGenerator) -> None:
    ctx = MessageContext(
        customer_id="DEMO04",
        customer_name="Rohit Patel",
        risk_score=65.0,
        risk_level=RiskLevel.MEDIUM,
        top_risk_factors=["failed_payments_count (↑)"],
        channel=Channel.APP,
        language=Language.EN,
        failed_payment_count=3,
        upcoming_emi_amount=12500,
        upcoming_emi_date="July 15",
    )
    msg = gen.generate(ctx)
    assert "12,500" in msg.body
    assert "July 15" in msg.body


def test_general_risk_fallback(gen: MessageGenerator) -> None:
    ctx = MessageContext(
        customer_id="TEST",
        customer_name="Test User",
        risk_score=50.0,
        risk_level=RiskLevel.MEDIUM,
        top_risk_factors=["unknown_factor"],
        channel=Channel.APP,
        language=Language.EN,
    )
    msg = gen.generate(ctx)
    assert msg.scenario == Scenario.GENERAL_RISK
