"""Behavioral regressions for matched rough/provider compaction calibration."""

import sqlite3
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from hermes_state import SessionDB


def _compressor(
    *,
    model: str = "gpt-5.6",
    provider: str = "openai-codex",
    base_url: str = "https://chatgpt.com/backend-api/codex",
    api_mode: str = "codex_responses",
    context_length: int = 372_000,
    max_tokens: int = 16_384,
) -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=context_length,
    ):
        return ContextCompressor(
            model=model,
            provider=provider,
            base_url=base_url,
            api_mode=api_mode,
            threshold_percent=0.80,
            max_tokens=max_tokens,
            quiet_mode=True,
        )


def _record_success(
    compressor: ContextCompressor,
    *,
    rough_tokens: int,
    prompt_tokens: int,
) -> None:
    compressor.begin_request_calibration(rough_tokens)
    compressor.update_from_response(
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 100,
            "total_tokens": prompt_tokens + 100,
        }
    )


def test_short_delta_uses_matched_provider_usage_instead_of_absolute_rough() -> None:
    compressor = _compressor()
    assert compressor.threshold_tokens == 284_492

    _record_success(compressor, rough_tokens=290_282, prompt_tokens=201_860)

    pressure = compressor.calibrated_pressure_tokens(290_458)
    assert pressure == 202_036
    assert compressor.should_compress(pressure) is False


def test_large_new_delta_can_cross_threshold() -> None:
    compressor = _compressor()
    _record_success(compressor, rough_tokens=290_282, prompt_tokens=201_860)

    pressure = compressor.calibrated_pressure_tokens(380_000)
    assert pressure == 291_578
    assert compressor.should_compress(pressure) is True


def test_real_usage_at_threshold_still_allows_compaction() -> None:
    compressor = _compressor()
    _record_success(
        compressor,
        rough_tokens=290_282,
        prompt_tokens=compressor.threshold_tokens,
    )

    pressure = compressor.calibrated_pressure_tokens(290_100)
    assert pressure == compressor.threshold_tokens
    assert compressor.should_compress(pressure) is True


def test_without_matched_pair_pressure_remains_conservative_absolute_rough() -> None:
    compressor = _compressor()

    assert compressor.calibrated_pressure_tokens(290_458) == 290_458
    assert compressor.should_compress(290_458) is True


def test_failed_request_estimate_is_not_paired_with_later_usage() -> None:
    compressor = _compressor()
    compressor.begin_request_calibration(290_282)
    compressor.discard_request_calibration()

    compressor.update_from_response(
        {"prompt_tokens": 201_860, "completion_tokens": 100, "total_tokens": 201_960}
    )

    assert compressor.calibrated_pressure_tokens(290_458) == 290_458


def test_request_identity_changes_invalidate_pair() -> None:
    changes = [
        {"model": "gpt-5.6-mini"},
        {"provider": "openai"},
        {"base_url": "https://relay.example/v1"},
        {"api_mode": "chat_completions"},
        {"context_length": 400_000},
        {"max_tokens": 8_192},
    ]

    for change in changes:
        compressor = _compressor()
        _record_success(compressor, rough_tokens=290_282, prompt_tokens=201_860)
        compressor.update_model(
            change.get("model", compressor.model),
            change.get("context_length", compressor.context_length),
            base_url=change.get("base_url", compressor.base_url),
            provider=change.get("provider", compressor.provider),
            api_mode=change.get("api_mode", compressor.api_mode),
            max_tokens=change.get("max_tokens", compressor.max_tokens),
        )

        assert compressor.calibrated_pressure_tokens(290_458) == 290_458


def test_persisted_identity_never_contains_raw_base_url_credentials(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-a", source="tui")
    compressor = _compressor(base_url="https://user:secret@example.invalid/v1")
    compressor.bind_session_state(db, "session-a")

    _record_success(compressor, rough_tokens=290_282, prompt_tokens=201_860)

    state = db.get_compaction_calibration("session-a")
    assert state is not None
    assert "secret" not in repr(state["identity"])
    assert "https://" not in repr(state["identity"])
    assert state["identity"]["base_url_sha256"]


def test_compaction_boundary_clears_persisted_pair(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-a", source="tui")
    compressor = _compressor()
    compressor.bind_session_state(db, "session-a")
    _record_success(compressor, rough_tokens=290_282, prompt_tokens=201_860)

    compressor.invalidate_matched_calibration()

    rebuilt = _compressor()
    rebuilt.bind_session_state(db, "session-a")
    assert rebuilt.calibrated_pressure_tokens(290_458) == 290_458
    assert db.get_compaction_calibration("session-a") is None


def test_matched_pair_round_trips_across_agent_rebuild_for_same_session(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("desktop-session", source="tui")

    first = _compressor()
    first.bind_session_state(db, "desktop-session")
    _record_success(first, rough_tokens=290_282, prompt_tokens=201_860)

    rebuilt = _compressor()
    rebuilt.bind_session_state(db, "desktop-session")

    assert rebuilt.calibrated_pressure_tokens(290_458) == 202_036


def test_persisted_pair_is_bound_to_runtime_identity_and_session(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-a", source="tui")
    db.create_session("session-b", source="tui")

    first = _compressor()
    first.bind_session_state(db, "session-a")
    _record_success(first, rough_tokens=290_282, prompt_tokens=201_860)

    wrong_model = _compressor(model="gpt-5.6-mini")
    wrong_model.bind_session_state(db, "session-a")
    other_session = _compressor()
    other_session.bind_session_state(db, "session-b")

    assert wrong_model.calibrated_pressure_tokens(290_458) == 290_458
    assert other_session.calibrated_pressure_tokens(290_458) == 290_458


def test_true_session_end_clears_persisted_pair(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-a", source="tui")
    compressor = _compressor()
    compressor.bind_session_state(db, "session-a")
    _record_success(compressor, rough_tokens=290_282, prompt_tokens=201_860)

    compressor.on_session_end("session-a", [])

    rebuilt = _compressor()
    rebuilt.bind_session_state(db, "session-a")
    assert rebuilt.calibrated_pressure_tokens(290_458) == 290_458
    assert db.get_compaction_calibration("session-a") is None


def test_session_db_calibration_state_is_explicit_and_clearable(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-a", source="tui")
    identity = {
        "model": "gpt-5.6",
        "provider": "openai-codex",
        "context_length": 372_000,
        "max_tokens": 16_384,
    }

    db.record_compaction_calibration(
        "session-a",
        identity=identity,
        rough_tokens=290_282,
        prompt_tokens=201_860,
    )
    assert db.get_compaction_calibration("session-a") == {
        "identity": identity,
        "rough_tokens": 290_282,
        "prompt_tokens": 201_860,
    }

    db.clear_compaction_calibration("session-a")
    assert db.get_compaction_calibration("session-a") is None


def test_existing_state_db_is_reconciled_before_calibration_round_trip(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("legacy-session", source="tui")
    db.close()

    conn = sqlite3.connect(db_path)
    for column in (
        "compaction_calibration_identity",
        "compaction_calibration_rough_tokens",
        "compaction_calibration_prompt_tokens",
    ):
        conn.execute(f"ALTER TABLE sessions DROP COLUMN {column}")
    conn.execute("UPDATE schema_version SET version = 20")
    conn.commit()
    conn.close()

    migrated = SessionDB(db_path=db_path)
    compressor = _compressor()
    compressor.bind_session_state(migrated, "legacy-session")
    _record_success(compressor, rough_tokens=290_282, prompt_tokens=201_860)

    rebuilt = _compressor()
    rebuilt.bind_session_state(migrated, "legacy-session")
    assert rebuilt.calibrated_pressure_tokens(290_458) == 202_036
