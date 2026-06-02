from __future__ import annotations

from collections import defaultdict

from censor_guard.schemas import ModerationResponse, ModerationRequest, SignalResult
from censor_guard.taxonomy import HARD_BLOCK_CATEGORIES, SOFT_REVIEW_CATEGORIES


class DecisionEngine:
    """Правило-ориентированный «сведённый вердикт».

    Берёт все сигналы сенсоров и для каждой категории нарушения вычисляет
    максимальную оценку (берётся самый «уверенный» сенсор). Затем сравнивает
    эти оценки с двумя порогами:
      - block_threshold  (по умолчанию 0.85) — выше него = блокировка;
      - review_threshold (по умолчанию 0.55) — выше него = ручная проверка.
    Hard-категории (sexual, violence_gore и т.д.) блокируются от любого сенсора;
    soft-категории блокируются только если их «поднял» policy judge.
    """

    def __init__(self, block_threshold: float, review_threshold: float) -> None:
        self.block_threshold = block_threshold
        self.review_threshold = review_threshold

    def decide(self, request: ModerationRequest, signals: list[SignalResult]) -> ModerationResponse:
        # category_scores  — макс. оценка по каждой категории среди всех сигналов.
        # category_sources — какие именно сигналы «увидели» эту категорию (для evidence).
        # judge_confirmed  — категории, подтверждённые хотя бы одним арбитром (role="judge").
        # notes            — причины пропусков/ошибок сенсоров (status != ok).
        category_scores: dict[str, float] = defaultdict(float)
        category_sources: dict[str, list[str]] = defaultdict(list)
        judge_confirmed: set[str] = set()
        notes: list[str] = []

        for signal in signals:
            # Пропущенные (skipped) и упавшие (error) сенсоры не голосуют —
            # только записываем их причину в notes для прозрачности ответа.
            if signal.status != "ok":
                if signal.reason:
                    notes.append(f"{signal.name}: {signal.reason}")
                continue
            for category, score in signal.categories.items():
                # Берём максимум: если хотя бы один сигнал уверен — категория «горит».
                if score >= category_scores[category]:
                    category_scores[category] = score
                if score > 0:
                    category_sources[category].append(signal.name)
                    # Подтверждение для soft-блока определяется ролью сигнала,
                    # а не его именем: любой арбитр (эвристика-консенсус или
                    # ShieldGemma), назвавший категорию, считается подтверждением.
                    if signal.role == "judge":
                        judge_confirmed.add(category)

        # Hard-категории блокируются от любого сигнала, перешагнувшего порог.
        hard_block_matches = [
            category
            for category in HARD_BLOCK_CATEGORIES
            if category_scores.get(category, 0.0) >= self.block_threshold
        ]
        # Soft-категории (harassment, политика и т.п.) — более «мягкие»: их
        # блокируем, только если высокую оценку подтвердил арбитр (role="judge"),
        # а не один сырой сенсор. Одиночный шумный сенсор в soft-категории так
        # дойдёт максимум до review, но не до block. Это и есть «гашение шума».
        soft_block_matches = [
            category
            for category in SOFT_REVIEW_CATEGORIES
            if category_scores.get(category, 0.0) >= self.block_threshold
            and category in judge_confirmed
        ]
        # Всё, что выше review-порога, но не дотянуло до блокировки → ручная проверка.
        review_matches = [
            category
            for category, score in category_scores.items()
            if score >= self.review_threshold and category not in hard_block_matches and category not in soft_block_matches
        ]

        if hard_block_matches or soft_block_matches:
            verdict = "block"
            categories = sorted(set(hard_block_matches + soft_block_matches))
            confidence = max(category_scores[category] for category in categories)
            reason = "Blocked by image guardrail due to unsafe content."
        elif review_matches:
            verdict = "review"
            categories = sorted(review_matches)
            confidence = max(category_scores[category] for category in categories)
            reason = "Request requires secondary review due to medium-confidence policy signals."
        else:
            verdict = "allow"
            categories = []
            confidence = max(category_scores.values(), default=0.0)
            reason = "No blocking policy signals exceeded review thresholds."

        evidence = {category: sorted(set(sources)) for category, sources in category_sources.items() if category in categories}
        return ModerationResponse(
            request_id=request.request_id,
            scenario=request.scenario,
            stage=request.stage,
            verdict=verdict,
            categories=categories,
            confidence=round(confidence, 4),
            reason=reason,
            evidence=evidence,
            signals=signals,
            notes=notes,
        )

