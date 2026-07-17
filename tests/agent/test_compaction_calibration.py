"""Behavioral regressions for matched rough/provider compaction calibration."""

import sqlite3
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.model_metadata import request_prompt_tool_fingerprint
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
    attempt_id = compressor.begin_request_calibration(rough_tokens)
    compressor.update_from_response(
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 100,
            "total_tokens": prompt_tokens + 100,
        },
        calibration_attempt_id=attempt_id,
    )


def _shape(label: str = "stable") -> str:
    return request_prompt_tool_fingerprint(
        {
            "messages": [{"role": "system", "content": f"prompt-{label}"}],
            "tools": [{"type": "function", "function": {"name": f"tool-{label}"}}],
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
    attempt_id = compressor.begin_request_calibration(290_282)
    compressor.discard_request_calibration(attempt_id)

    compressor.update_from_response(
        {"prompt_tokens": 201_860, "completion_tokens": 100, "total_tokens": 201_960}
    )

    assert compressor.calibrated_pressure_tokens(290_458) == 290_458


def test_out_of_order_responses_pair_only_with_their_exact_attempt() -> None:
    compressor = _compressor()
    first = compressor.begin_request_calibration(100_000)
    second = compressor.begin_request_calibration(200_000)

    compressor.update_from_response(
        {"prompt_tokens": 120_000}, calibration_attempt_id=second
    )
    compressor.update_from_response(
        {"prompt_tokens": 60_000}, calibration_attempt_id=first
    )

    assert compressor.matched_request_rough_tokens == 200_000
    assert compressor.matched_prompt_tokens == 120_000


def test_two_compressors_sharing_session_reject_older_completion_last(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("shared", source="tui")
    older = _compressor()
    newer = _compressor()
    older.bind_session_state(db, "shared")
    newer.bind_session_state(db, "shared")

    old_attempt = older.begin_request_calibration(100_000)
    new_attempt = newer.begin_request_calibration(200_000)
    newer.update_from_response(
        {"prompt_tokens": 120_000}, calibration_attempt_id=new_attempt
    )
    older.update_from_response(
        {"prompt_tokens": 60_000}, calibration_attempt_id=old_attempt
    )

    state = db.get_compaction_calibration("shared")
    assert state is not None
    assert state["rough_tokens"] == 200_000
    assert state["prompt_tokens"] == 120_000
    assert state["pair_sequence"] == new_attempt


def test_effective_output_cap_is_attempt_identity_not_configured_default() -> None:
    compressor = _compressor(max_tokens=16_384)
    stable_shape = _shape()
    compressor.set_request_calibration_shape(stable_shape, effective_output_cap=16_384)
    normal = compressor.begin_request_calibration(
        100_000,
        prompt_tool_fingerprint=stable_shape,
        effective_output_cap=16_384,
    )
    compressor.update_from_response(
        {"prompt_tokens": 70_000}, calibration_attempt_id=normal
    )

    retry = compressor.begin_request_calibration(
        101_000,
        prompt_tool_fingerprint=stable_shape,
        effective_output_cap=32_768,
    )
    compressor.update_from_response(
        {"prompt_tokens": 71_000}, calibration_attempt_id=retry
    )

    compressor.set_request_calibration_shape(stable_shape, effective_output_cap=16_384)
    assert compressor.calibrated_pressure_tokens(102_000) == 102_000


def test_retry_output_cap_changes_effective_compaction_threshold() -> None:
    compressor = _compressor(context_length=1_000_000, max_tokens=10_000)

    assert compressor.should_compress_for_output_cap(600_000, 10_000) is False
    assert compressor.should_compress_for_output_cap(600_000, 300_000) is True


def test_prompt_or_tool_fingerprint_change_invalidates_pair_and_durable_row(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-a", source="tui")
    compressor = _compressor()
    compressor.bind_session_state(db, "session-a")
    original = _shape("original")
    changed = _shape("changed")
    compressor.set_request_calibration_shape(original, effective_output_cap=16_384)
    attempt = compressor.begin_request_calibration(
        100_000,
        prompt_tool_fingerprint=original,
        effective_output_cap=16_384,
    )
    compressor.update_from_response(
        {"prompt_tokens": 70_000}, calibration_attempt_id=attempt
    )

    compressor.set_request_calibration_shape(changed, effective_output_cap=16_384)

    assert compressor.calibrated_pressure_tokens(101_000) == 101_000
    assert db.get_compaction_calibration("session-a") is None


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


def test_persisted_identity_contains_hashes_not_prompt_or_tool_contents(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-a", source="tui")
    compressor = _compressor()
    compressor.bind_session_state(db, "session-a")
    fingerprint = request_prompt_tool_fingerprint(
        {
            "messages": [{"role": "system", "content": "TOP SECRET PROMPT"}],
            "tools": [{"type": "function", "function": {"name": "secret_tool"}}],
        }
    )
    compressor.set_request_calibration_shape(fingerprint, effective_output_cap=16_384)
    attempt = compressor.begin_request_calibration(
        100_000,
        prompt_tool_fingerprint=fingerprint,
        effective_output_cap=16_384,
    )
    compressor.update_from_response(
        {"prompt_tokens": 70_000}, calibration_attempt_id=attempt
    )

    state = db.get_compaction_calibration("session-a")
    assert state is not None
    persisted = repr(state["identity"])
    assert "TOP SECRET PROMPT" not in persisted
    assert "secret_tool" not in persisted
    assert state["identity"]["prompt_tool_sha256"] == fingerprint
    assert state["identity"]["estimator_version"] > 0


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


def test_shaped_pair_survives_rebuild_until_prompt_tools_are_known(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("desktop-session", source="tui")
    fingerprint = _shape("restored")
    first = _compressor()
    first.bind_session_state(db, "desktop-session")
    first.set_request_calibration_shape(fingerprint, effective_output_cap=16_384)
    attempt = first.begin_request_calibration(100_000)
    first.update_from_response(
        {"prompt_tokens": 70_000}, calibration_attempt_id=attempt
    )

    rebuilt = _compressor()
    rebuilt.bind_session_state(db, "desktop-session")
    assert db.get_compaction_calibration("desktop-session") is not None
    rebuilt.set_request_calibration_shape(fingerprint, effective_output_cap=16_384)
    assert rebuilt.calibrated_pressure_tokens(101_000) == 71_000


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


def test_restore_identity_mismatch_clears_stale_row(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-a", source="tui")
    first = _compressor()
    first.bind_session_state(db, "session-a")
    _record_success(first, rough_tokens=100_000, prompt_tokens=70_000)

    mismatched = _compressor(model="gpt-5.6-mini")
    mismatched.bind_session_state(db, "session-a")

    assert db.get_compaction_calibration("session-a") is None


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
        attempt_sequence=db.begin_compaction_calibration_attempt("session-a"),
        identity=identity,
        rough_tokens=290_282,
        prompt_tokens=201_860,
    )
    assert db.get_compaction_calibration("session-a") == {
        "identity": identity,
        "rough_tokens": 290_282,
        "prompt_tokens": 201_860,
        "pair_sequence": 1,
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
        "compaction_calibration_attempt_sequence",
        "compaction_calibration_pair_sequence",
    ):
        conn.execute(f"ALTER TABLE sessions DROP COLUMN {column}")
    conn.commit()
    conn.close()

    migrated = SessionDB(db_path=db_path)
    compressor = _compressor()
    compressor.bind_session_state(migrated, "legacy-session")
    _record_success(compressor, rough_tokens=290_282, prompt_tokens=201_860)

    rebuilt = _compressor()
    rebuilt.bind_session_state(migrated, "legacy-session")
    assert rebuilt.calibrated_pressure_tokens(290_458) == 202_036


def test_append_style_transcript_replacement_preserves_calibration(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-a", source="tui")
    db.append_message("session-a", "user", "first")
    compressor = _compressor()
    compressor.bind_session_state(db, "session-a")
    _record_success(compressor, rough_tokens=100_000, prompt_tokens=70_000)

    db.replace_messages(
        "session-a",
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "new answer"},
        ],
    )
    assert db.get_compaction_calibration("session-a") is not None


def test_structural_rewrite_and_rewind_clear_calibration(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-a", source="tui")
    db.append_message("session-a", "user", "first")
    db.append_message("session-a", "assistant", "answer")
    compressor = _compressor()
    compressor.bind_session_state(db, "session-a")
    _record_success(compressor, rough_tokens=100_000, prompt_tokens=70_000)

    db.replace_messages("session-a", [{"role": "user", "content": "retry"}])
    assert db.get_compaction_calibration("session-a") is None

    rebuilt = _compressor()
    rebuilt.bind_session_state(db, "session-a")
    _record_success(rebuilt, rough_tokens=110_000, prompt_tokens=75_000)
    target = db.list_recent_user_messages("session-a", limit=1)[0]["id"]
    db.rewind_to_message("session-a", target)
    assert db.get_compaction_calibration("session-a") is None
