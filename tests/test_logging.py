"""Tests for logging configuration."""

from __future__ import annotations

import json
import logging

from ev_battery.logging_config import JSONFormatter, setup_logging


class TestJSONFormatter:
    """JSONFormatter should produce valid JSON log lines."""

    def test_basic_log_is_valid_json(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__,
            lineno=1, msg="hello world", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello world"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"
        assert "timestamp" in parsed

    def test_extra_fields_included(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname=__file__,
            lineno=1, msg="warned", args=(), exc_info=None,
        )
        record.battery_id = "B001"  # type: ignore[attr-defined]
        record.soc = 0.85  # type: ignore[attr-defined]
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["battery_id"] == "B001"
        assert parsed["soc"] == 0.85

    def test_exception_info_included(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname=__file__,
                lineno=1, msg="error occurred", args=(), exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestSetupLogging:
    """setup_logging should configure the root logger."""

    def test_sets_level(self):
        setup_logging("DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_replaces_handlers(self):
        setup_logging("INFO")
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)

    def test_json_output(self, caplog: pytest.LogCaptureFixture):
        setup_logging("INFO")
        logger = logging.getLogger("test_json")
        logger.info("test message", extra={"battery_id": "B002"})
        # The last log record should be JSON-parseable
        # (caplog captures the formatted output via a handler on the root)
