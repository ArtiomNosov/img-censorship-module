from __future__ import annotations

from censor_guard.adapters.explicit_detector import ExplicitContentAdapter
from censor_guard.adapters.policy_judge import PolicyJudgeFacade
from censor_guard.adapters.visual_classifier import VisualClassifierAdapter
from censor_guard.config import Settings
from censor_guard.decision import DecisionEngine
from censor_guard.image_utils import load_image
from censor_guard.schemas import ModerationRequest, ModerationResponse


class GuardrailPipeline:
    """Оркестратор конвейера модерации (алгоритм v2).

    Создаёт все адаптеры-«сенсоры» один раз (в __init__) и переиспользует их
    между запросами. Тяжёлые ML-модели внутри адаптеров грузятся лениво —
    только при первом реальном вызове, поэтому сам по себе конструктор дешёвый.
    Метод moderate() прогоняет один запрос через все сенсоры и сводит их
    сигналы в итоговый вердикт через DecisionEngine.

    В v2 конвейер сфокусирован на изображении: работают визуальный
    zero-shot классификатор, NSFW-детектор и арбитр (ShieldGemma + консенсус-
    эвристика). Текстовый гард и OCR намеренно отключены на этом этапе
    (см. PLAN.md, P0.1) — контракты адаптеров остаются в коде для будущего.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.visual = VisualClassifierAdapter(
            enabled=self.settings.enable_visual_classifier,
            model_id=self.settings.visual_model_id,
            cache_dir=self.settings.hf_cache_dir,
        )
        self.explicit = ExplicitContentAdapter(
            enabled=self.settings.enable_explicit_detector,
            model_id=self.settings.explicit_model_id,
            cache_dir=self.settings.hf_cache_dir,
        )
        self.policy_judge = PolicyJudgeFacade(
            enabled=self.settings.enable_policy_judge,
            model_id=self.settings.policy_judge_model_id,
            cache_dir=self.settings.hf_cache_dir,
        )
        self.decision_engine = DecisionEngine(
            block_threshold=self.settings.block_threshold,
            review_threshold=self.settings.review_threshold,
        )

    def moderate(self, request: ModerationRequest) -> ModerationResponse:
        # Каждый сенсор возвращает SignalResult — единый «конверт» с категориями
        # и их оценками. Мы собираем их в список signals и в конце сводим воедино.
        image = load_image(request)
        if image is None:
            # v2 анализирует изображение; без него модерировать пока нечего
            # (текстовый гард отложен — см. PLAN.md, P0.1). Возвращаем нейтральный
            # ответ через движок с пустым набором сигналов.
            return self.decision_engine.decide(request, [])

        # 1) Два визуальных сенсора: zero-shot мульти-классификатор по нашей
        #    таксономии и узкоспециализированный детектор NSFW.
        signals = [
            self.visual.moderate(image),
            self.explicit.moderate(image),
        ]

        # 2) Арбитры. ShieldGemma (если включена) даёт сильный независимый сигнал
        #    по hard-категориям; консенсус-эвристика подтверждает soft-категории,
        #    которые увидели несколько сенсоров. Оба идут с role="judge".
        signals.extend(
            self.policy_judge.moderate(
                image=image,
                prompt=request.prompt,
                signals=signals,
            )
        )

        # 3) Финальное решение allow / review / block по порогам.
        return self.decision_engine.decide(request, signals)
