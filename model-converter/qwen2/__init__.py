from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available
)

_import_structure = {
    "configuration_qwen2": ["Qwen2Config"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_qwen2"] = [
        "Qwen2ForCausalLM",
        "Qwen2ForQuestionAnswering",
        "Qwen2Model",
        "Qwen2PreTrainedModel",
        "Qwen2ForSequenceClassification",
        "Qwen2ForTokenClassification",
    ]

if TYPE_CHECKING:
    from .configuration_qwen2 import Qwen2Config

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_qwen2 import (
            Qwen2ForCausalLM,
            Qwen2ForQuestionAnswering,
            Qwen2Model,
            Qwen2PreTrainedModel,
            Qwen2ForSequenceClassification,
            Qwen2ForTokenClassification,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)