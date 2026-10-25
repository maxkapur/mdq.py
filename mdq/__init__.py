from onnxruntime import set_default_logger_severity
from rich.console import Console

# Suppress warning about CPU-only mode
set_default_logger_severity(4)

console = Console(stderr=True)
