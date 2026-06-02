from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from censor_guard.schemas import SignalResult
from censor_guard.taxonomy import CATEGORY_SPECS


class HeuristicPolicyJudge:
    """Эвристический «арбитр» на случай, когда настоящей модели-судьи нет.

    Его единственная задача — давать DecisionEngine *подтверждение* (gate) для
    soft-категорий. Идея простая: одиночному сырому сенсору в soft-категории
    доверять нельзя (риск ложной блокировки), а вот согласие нескольких
    независимых сенсоров — уже сигнал.

    Поэтому судья **подтверждает категорию только при консенсусе ≥2 сенсоров**.
    Эмитит он при этом не глобальный максимум (его и так считает DecisionEngine —
    дублировать незачем), а *вторую по величине* оценку — тот уровень, на котором
    согласны как минимум два сенсора. Это самостоятельная, осмысленная величина,
    которая никогда не превышает глобальный max, поэтому не «накручивает» фьюжн.

    Категории, которые увидел лишь один сенсор, судья намеренно не подтверждает:
    для soft-категорий это значит, что они дойдут максимум до review, но не до
    block. Hard-категориям подтверждение судьи не требуется вовсе (их блокирует
    любой сенсор), поэтому их консенсус здесь не считаем.
    """

    name = "policy_judge_heuristic"
    consensus_min_sensors = 2

    def moderate(self, signals: list[SignalResult]) -> SignalResult:
        # Собираем по каждой категории оценки независимых СЕНСОРОВ (role="sensor").
        # Другие арбитры (role="judge") здесь не учитываются — консенсус считается
        # только среди первичных детекторов.
        per_category_scores: dict[str, list[float]] = defaultdict(list)
        for signal in signals:
            if signal.status != "ok" or signal.role != "sensor":
                continue
            for category, score in signal.categories.items():
                if score > 0:
                    per_category_scores[category].append(score)

        confirmed: dict[str, float] = {}
        for category, scores in per_category_scores.items():
            if len(scores) >= self.consensus_min_sensors:
                scores.sort(reverse=True)
                # Вторая по величине оценка = уровень, на котором согласны ≥2 сенсора.
                confirmed[category] = scores[1]

        return SignalResult(
            name=self.name,
            status="ok",
            role="judge",
            categories=confirmed,
            reason="Heuristic judge confirmed categories with multi-sensor consensus.",
            raw={
                "mode": "heuristic-consensus",
                "evidence_count": {c: len(s) for c, s in per_category_scores.items()},
            },
        )


class ShieldGemmaJudge:
    """Настоящий мультимодальный арбитр на базе ShieldGemma-2.

    Загружается лениво (как и остальные ML-адаптеры): тяжёлая модель
    инициализируется при первом реальном вызове и далее переиспользуется.
    Любой сбой превращается в «мягкую» деградацию (skipped/error), а не в падение
    сервиса — тогда конвейер опирается на HeuristicPolicyJudge.

    Мы спрашиваем модель по нашим hard-категориям (их описания из taxonomy
    отдаются ShieldGemma как кастомные политики) и кладём вероятность нарушения
    в соответствующий код категории. Это независимый и более надёжный сигнал,
    чем zero-shot, поэтому он участвует и во фьюжне, и как подтверждение.
    """

    name = "policy_judge_shieldgemma"

    def __init__(self, enabled: bool, model_id: str, cache_dir: str) -> None:
        self.enabled = enabled
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None
        self._torch = None
        self._load_error: str | None = None
        # Спрашиваем ShieldGemma по hard-категориям; soft-категории закрывает
        # консенсус-эвристика. Порядок кодов фиксирован — по нему разбираем выход.
        self._policy_codes = [spec.code for spec in CATEGORY_SPECS if spec.hard_block]
        self._custom_policies = {
            spec.code: spec.description for spec in CATEGORY_SPECS if spec.hard_block
        }

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import torch
            from transformers import AutoProcessor, ShieldGemma2ForImageClassification
        except ImportError:
            return False
        try:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", self.cache_dir)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cache_dir)
            os.environ.setdefault("TRANSFORMERS_CACHE", self.cache_dir)
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = ShieldGemma2ForImageClassification.from_pretrained(
                self.model_id
            ).eval()
            self._torch = torch
        except Exception as exc:  # pragma: no cover - backend-specific failures
            self._load_error = str(exc)
            return False
        return True

    def moderate(self, image, prompt: str | None = None) -> SignalResult:
        if not self.enabled:
            return SignalResult(
                name=self.name,
                status="skipped",
                role="judge",
                reason="ShieldGemma judge disabled by configuration.",
            )
        if image is None:
            return SignalResult(
                name=self.name,
                status="skipped",
                role="judge",
                reason="ShieldGemma judge needs an image; none supplied.",
            )
        if not self._load():
            if self._load_error:
                return SignalResult(
                    name=self.name,
                    status="error",
                    role="judge",
                    reason=f"ShieldGemma judge load failed: {self._load_error}",
                )
            return SignalResult(
                name=self.name,
                status="skipped",
                role="judge",
                reason="transformers/torch is not installed for ShieldGemma.",
            )

        try:
            inputs = self._processor(
                images=[image],
                custom_policies=self._custom_policies,
                policies=self._policy_codes,
                return_tensors="pt",
            ).to(self._model.device)
            with self._torch.inference_mode():
                output = self._model(**inputs)
            # output.probabilities: тензор (num_images * num_policies, 2). По коду
            # модели logits берутся как [yes_token_index, no_token_index], значит
            # колонка 0 — вероятность токена «Yes» (= политика НАРУШЕНА), колонка 1
            # — «No» (безопасно). Строки идут в порядке policies: img1_p1..img1_pN.
            probabilities = output.probabilities.detach().to("cpu").tolist()
        except Exception as exc:  # pragma: no cover - backend-specific failures
            return SignalResult(
                name=self.name,
                status="error",
                role="judge",
                reason=f"ShieldGemma judge inference failed: {exc}",
            )

        categories: dict[str, float] = {}
        raw: dict[str, float] = {}
        for idx, code in enumerate(self._policy_codes):
            if idx >= len(probabilities):
                break
            violation_prob = float(probabilities[idx][0])
            raw[code] = violation_prob
            if violation_prob > 0:
                categories[code] = violation_prob

        return SignalResult(
            name=self.name,
            status="ok",
            role="judge",
            categories=categories,
            reason="ShieldGemma evaluated image against hard-block policies.",
            raw={"model_id": self.model_id, "violation_probabilities": raw},
        )


class PolicyJudgeFacade:
    """Фасад над арбитрами. В отличие от MVP здесь это не «или-или»:

    - HeuristicPolicyJudge запускается **всегда** — он даёт консенсус-подтверждение
      для всех (в т.ч. soft) категорий, опираясь на дешёвые сенсоры;
    - ShieldGemmaJudge добавляется как сильный независимый сигнал по hard-категориям,
      когда модель доступна (`CENSOR_ENABLE_POLICY_JUDGE=true`).

    Возвращает список сигналов — пайплайн просто добавляет их к остальным.
    """

    def __init__(self, enabled: bool, model_id: str, cache_dir: str) -> None:
        self.shieldgemma = ShieldGemmaJudge(
            enabled=enabled, model_id=model_id, cache_dir=cache_dir
        )
        self.heuristic = HeuristicPolicyJudge()

    def moderate(
        self, image, prompt: str | None, signals: list[SignalResult]
    ) -> list[SignalResult]:
        return [
            self.shieldgemma.moderate(image=image, prompt=prompt),
            self.heuristic.moderate(signals),
        ]
