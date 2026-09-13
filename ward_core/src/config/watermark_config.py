from enum import Enum
from typing import List

from pydantic import Field

from src.config.utils import PydanticBaseModelWithOptionalDefaultsPath as PBMwODP


class WatermarkDetectionNormalizer(Enum):
    UNICODE = "unicode"
    HOMOGLYPHS = "homoglyphs"
    TRUECASE = "truecase"


class WatermarkScheme(Enum):
    KGW = "kgw"


class MetaConfig(PBMwODP, extra="forbid"):  # type: ignore
    device: str = Field(..., description="Device to run on (cuda/cpu)")
    seed: int = Field(..., description="Random seed")
    out_root_dir: str = Field(..., description="Directory to save outputs to")
    result_dir: str = Field("results/default", description="Directory to save evaluation results to")

    def short_str(self) -> str:
        return f"seed={self.seed}"


class ModelConfig(PBMwODP, arbitrary_types_allowed=True):  # type: ignore
    skip: bool = Field(..., description="If this model should be loaded or skipped")
    name: str = Field(description="Model name or path")
    use_fp16: bool = Field(False, description="Load and use FP16 precision")
    use_flashattn2: bool = Field(False, description="Use flash attention when supported")
    prompt_max_len: int = Field(None, description="Max length of the prompt")
    response_max_len: int = Field(None, description="Max length of the response")
    n_beams: int = Field(1, description="Number of beams for beam search")
    use_sampling: bool = Field(False, description="Use sampling instead of greedy decoding")
    sampling_temp: float = Field(0.7, description="Sampling temperature")

    def short_str(self) -> str:
        return f"name={self.name},n_beams={self.n_beams},sample={self.use_sampling},temp={self.sampling_temp}"


class WatermarkGenerationConfig(PBMwODP, extra="forbid"):  # type: ignore
    seeding_scheme: str = Field(..., description="KGW seeding scheme")
    gamma: float = Field(..., description="Fraction of green tokens")
    delta: float = Field(..., description="Logit boost for green tokens")


class WatermarkDetectionConfig(PBMwODP, extra="forbid"):  # type: ignore
    normalizers: List[WatermarkDetectionNormalizer] = Field(..., description="Detection normalizers")
    ignore_repeated_ngrams: bool = Field(..., description="Ignore repeated ngrams in detection")
    z_threshold: float = Field(..., description="Minimum z-score to mark text as watermarked")


class WatermarkConfig(PBMwODP, extra="forbid"):  # type: ignore
    scheme: WatermarkScheme = Field(..., description="Watermark scheme to use")
    generation: WatermarkGenerationConfig
    detection: WatermarkDetectionConfig
