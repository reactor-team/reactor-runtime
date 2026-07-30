"""The engine/runtime contract — a small shared vocabulary, owned by neither side.

An inference engine declares its input surface and satisfies one protocol; a
runtime reads those declarations and serves them. This package is what both
import, and it imports nothing else: no runtime, no transport, no serving code.
Treat its packaging as swappable — it is a vocabulary, not a framework.

An engine declares three things and nothing more::

    from reactor_runtime.engine_contract import Init, InputField, ModelInput, UserInput

    class Move(UserInput):
        direction: Literal["forward", "back", "left", "right"]
        speed: float = InputField(default=1.0, ge=0.0, le=4.0)

    class WalkInit(Init):
        prompt: str = "a walk through a misty forest"

    class WalkStepInput(ModelInput):
        trajectory: Any

    class WalkPipeline:                      # satisfies StreamingPipeline
        def initialize_cache(self, **init): ...
        def get_num_output_frames(self, autoregressive_index): ...
        def map_inputs(self, autoregressive_index, cache, inputs): ...
        def generate(self, autoregressive_index, cache, input=None): ...
        def finalize(self, autoregressive_index, cache): ...

Every input the client sends is queued and stamped on arrival, and each step's
``map_inputs`` receives every input since the previous step, ordered by
timestamp. That single ordered window is the whole input rule: there are no
per-parameter aggregation modes, and a value that has to persist between steps
is rollout state that lives on the cache.
"""

from reactor_runtime.engine_contract.fields import NO_DEFAULT, FieldSpec, InputField
from reactor_runtime.engine_contract.inputs import (
    USER_INPUT_REGISTRY,
    AudioInput,
    Init,
    MediaInput,
    ModelInput,
    UserInput,
    VideoInput,
)
from reactor_runtime.engine_contract.pipeline import Cache, StreamingPipeline, VideoChunk

__all__ = [
    "NO_DEFAULT",
    "USER_INPUT_REGISTRY",
    "AudioInput",
    "Cache",
    "FieldSpec",
    "Init",
    "InputField",
    "MediaInput",
    "ModelInput",
    "StreamingPipeline",
    "UserInput",
    "VideoChunk",
    "VideoInput",
]
