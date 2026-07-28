from robata.queue.backpressure import (
    BackpressureConfig,
    BackpressureController,
    BackpressureControllerMode,
    BackpressureControllerState,
    PressureClass,
    QueueMetrics,
)
from robata.queue.stage import Stage


def _controller(
    *,
    mode: BackpressureControllerMode = BackpressureControllerMode.FIXED,
    cooldown_ms: int = 0,
) -> BackpressureController:
    return BackpressureController(
        BackpressureConfig(
            version="test-backpressure-policy-v1",
            queue_depth_threshold=16,
            oldest_age_threshold_ms=1_000,
            backlog_slope_threshold=8.0,
            controller_mode=mode,
            minimum_limit=2,
            maximum_limit=16,
            additive_increase=2,
            multiplicative_decrease=0.5,
            cooldown_ms=cooldown_ms,
        )
    )


def test_unknown_signals_are_not_silently_measured_as_zero() -> None:
    controller = _controller()
    metrics = QueueMetrics(depth=0, oldest_age_ms=0)

    decision = controller.should_admit(Stage.QA_COARSE_PLAN, metrics)

    assert metrics.observation_version == "queue-metrics-v2"
    assert metrics.arrival_rate is None
    assert metrics.service_rate is None
    assert metrics.backlog_slope is None
    assert metrics.provider_quota is None
    assert decision.admitted
    assert decision.pressure_class is PressureClass.NORMAL


def test_signed_drain_slope_and_explicit_quota_have_distinct_meanings() -> None:
    controller = _controller()
    draining = QueueMetrics(depth=3, oldest_age_ms=0, backlog_slope=-2.5)
    quota_exhausted = QueueMetrics(depth=0, oldest_age_ms=0, provider_quota=0)

    assert controller.should_admit(Stage.QA_COARSE_PLAN, draining).admitted
    throttled = controller.should_admit(Stage.QA_COARSE_PLAN, quota_exhausted)
    assert not throttled.admitted
    assert throttled.signals == ("PROVIDER_QUOTA",)


def test_aimd_state_is_deterministic_across_restart_and_cooldown() -> None:
    controller = _controller(mode=BackpressureControllerMode.AIMD, cooldown_ms=100)
    state = controller.initial_state("provider-a")
    saturated = QueueMetrics(depth=0, oldest_age_ms=0, provider_quota=0)
    normal = QueueMetrics(depth=0, oldest_age_ms=0)

    throttled, after_throttle = controller.evaluate(
        Stage.QA_COARSE_PLAN,
        saturated,
        state,
        observed_at_ms=0,
    )
    held, after_hold = controller.evaluate(
        Stage.QA_COARSE_PLAN,
        normal,
        after_throttle,
        observed_at_ms=50,
    )
    increased, after_increase = controller.evaluate(
        Stage.QA_COARSE_PLAN,
        normal,
        after_hold,
        observed_at_ms=150,
    )

    assert throttled.controller_limit == 8
    assert held.controller_limit == 8
    assert increased.controller_limit == 10

    restarted = BackpressureControllerState.model_validate_json(after_hold.model_dump_json())
    replayed, replayed_state = controller.evaluate(
        Stage.QA_COARSE_PLAN,
        normal,
        restarted,
        observed_at_ms=150,
    )
    assert replayed == increased
    assert replayed_state == after_increase
