from __future__ import annotations

import unittest

from censor_guard.adapters.policy_judge import HeuristicPolicyJudge
from censor_guard.schemas import SignalResult


class HeuristicPolicyJudgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.judge = HeuristicPolicyJudge()

    def test_confirms_only_consensus_categories(self) -> None:
        # Две независимые сенсорные оценки одной категории → консенсус → подтверждение.
        signals = [
            SignalResult(name="visual_classifier", status="ok", categories={"sexual": 0.9}),
            SignalResult(name="explicit_content_detector", status="ok", categories={"sexual": 0.7}),
        ]
        result = self.judge.moderate(signals)
        self.assertEqual(result.role, "judge")
        self.assertIn("sexual", result.categories)
        # Эмитим вторую по величине оценку (уровень согласия), а не глобальный max.
        self.assertAlmostEqual(result.categories["sexual"], 0.7)

    def test_single_sensor_not_confirmed(self) -> None:
        signals = [
            SignalResult(name="visual_classifier", status="ok", categories={"political_persuasion": 0.95}),
        ]
        result = self.judge.moderate(signals)
        self.assertNotIn("political_persuasion", result.categories)

    def test_ignores_non_sensor_and_non_ok_signals(self) -> None:
        # Сигналы других арбитров (role="judge") и неуспешные в консенсусе не участвуют.
        signals = [
            SignalResult(name="visual_classifier", status="ok", categories={"sexual": 0.9}),
            SignalResult(name="other_judge", status="ok", role="judge", categories={"sexual": 0.9}),
            SignalResult(name="explicit_content_detector", status="error", categories={"sexual": 0.9}),
        ]
        result = self.judge.moderate(signals)
        # Только один настоящий сенсор увидел категорию → консенсуса нет.
        self.assertNotIn("sexual", result.categories)


if __name__ == "__main__":
    unittest.main()
