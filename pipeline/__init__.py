# pipeline/__init__.py
"""
Modular Pipeline Package
========================
Stage 1: SegFormer         → water_mask
Stage 2: YOLOv8            → reference_objects
Stage 3: Depth Anything V2 → depth_map
Stage 4: Fusion Engine     → structured_features
Stage 5: Severity Model    → FloodOutput
"""
from .segformer   import SegFormerStage
from .yolo        import YOLOStage
from .depth       import DepthStage
from .fusion      import FusionStage
from .severity    import SeverityStage
from .runner      import PipelineRunner
