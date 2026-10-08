"""Policy-model exports with lazy imports for optional language dependencies."""

from importlib import import_module


_EXPORTS = {
    "LANGUAGE_MODES": ("models.language_bc", "LANGUAGE_MODES"),
    "POLICY_CAMERAS": ("models.language_bc", "POLICY_CAMERAS"),
    "VISUAL_FUSION_MODES": ("models.language_bc", "VISUAL_FUSION_MODES"),
    "LanguageBCPolicy": ("models.language_bc", "LanguageBCPolicy"),
    "ResNet18VisualEncoder": ("models.visual_encoder", "ResNet18VisualEncoder"),
    "VISUAL_BC_FUSION_MODES": ("models.visual_bc", "VISUAL_BC_FUSION_MODES"),
    "LearnedSpatialAttention": ("models.visual_bc", "LearnedSpatialAttention"),
    "HistoryVisualBCPolicy": ("models.visual_bc", "HistoryVisualBCPolicy"),
    "VisualBCPolicy": ("models.visual_bc", "VisualBCPolicy"),
    "SpatialSoftmaxVisualBCPolicy": (
        "models.spatial_softmax_visual_bc",
        "SpatialSoftmaxVisualBCPolicy",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
