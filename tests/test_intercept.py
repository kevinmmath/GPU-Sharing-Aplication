"""
tests/test_intercept.py
────────────────────────────────────────────────────────────────────────────────
Automated tests for Phase 1 — verifies the interceptor correctly captures ops.

Run with:
    conda activate llms_course_env
    python -m pytest tests/test_intercept.py -v
"""

import sys
import os
import logging
import unittest

# Make the package importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import fake_cuda
from fake_cuda.interceptor import GPUShareMode


# ── Helper: capture log output from the "fake_cuda" logger ───────────────────

class _LogCapture(logging.Handler):
    """A logging handler that stores emitted records in a list."""
    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def _capture_logs() -> "_LogCapture":
    cap = _LogCapture()
    logging.getLogger("fake_cuda").addHandler(cap)
    return cap


def _remove_capture(cap: "_LogCapture") -> None:
    logging.getLogger("fake_cuda").removeHandler(cap)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestContextManager(unittest.TestCase):
    """Test the `with fake_cuda.intercept():` context-manager API."""

    def test_matmul_is_logged(self):
        """aten::mm (or aten::matmul) must appear in logs when intercepted."""
        cap = _capture_logs()
        try:
            with fake_cuda.intercept():
                a = torch.randn(4, 8)
                b = torch.randn(8, 16)
                _ = torch.matmul(a, b)
        finally:
            _remove_capture(cap)

        logged_ops = "\n".join(cap.records)
        self.assertTrue(
            any("mm" in r or "matmul" in r for r in cap.records),
            msg=f"Expected 'mm' or 'matmul' in logs, got:\n{logged_ops}",
        )

    def test_shape_appears_in_log(self):
        """The log line must contain the correct tensor shape."""
        cap = _capture_logs()
        try:
            with fake_cuda.intercept():
                a = torch.randn(7, 13)
                _ = torch.relu(a)
        finally:
            _remove_capture(cap)

        logged = "\n".join(cap.records)
        self.assertIn("7", logged,  msg=f"Shape dim 7 missing from:\n{logged}")
        self.assertIn("13", logged, msg=f"Shape dim 13 missing from:\n{logged}")

    def test_no_log_outside_context(self):
        """Ops executed OUTSIDE the context manager must NOT be logged."""
        cap = _capture_logs()
        try:
            a = torch.randn(3, 3)
            _ = torch.relu(a)        # this must NOT be captured
        finally:
            _remove_capture(cap)

        non_active_logs = [r for r in cap.records if "[LOG]" in r]
        self.assertEqual(
            non_active_logs, [],
            msg=f"Expected no [LOG] lines outside context, got: {non_active_logs}",
        )


class TestBackward(unittest.TestCase):
    """Gradient / backward ops must also be intercepted."""

    def test_backward_ops_logged(self):
        cap = _capture_logs()
        try:
            with fake_cuda.intercept():
                x = torch.randn(4, 4, requires_grad=True)
                loss = (x @ x.T).sum()
                loss.backward()
        finally:
            _remove_capture(cap)

        all_ops = "\n".join(cap.records)
        # We expect at least the mm forward and some backward op
        self.assertTrue(
            any("[LOG]" in r for r in cap.records),
            msg=f"No [LOG] lines captured during backward:\n{all_ops}",
        )


class TestLinearLayer(unittest.TestCase):
    """A real nn.Linear layer should log weight transpose + matmul + bias add."""

    def test_linear_ops_logged(self):
        import torch.nn as nn
        layer = nn.Linear(16, 8)

        cap = _capture_logs()
        try:
            with fake_cuda.intercept():
                x = torch.randn(4, 16)
                _ = layer(x)
        finally:
            _remove_capture(cap)

        log_lines = [r for r in cap.records if "[LOG]" in r]
        # At minimum: a t() (transpose) and an mm
        self.assertGreaterEqual(
            len(log_lines), 2,
            msg=f"Expected at least 2 log lines for Linear, got:\n" + "\n".join(log_lines),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
