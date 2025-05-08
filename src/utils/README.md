# TEMPO Utilities

This directory contains utility modules used throughout the TEMPO project.

## Debugging and Logging

### Configuration

The debug mode for all modules can be configured in several ways:

1. **Global setting**: Set the `TEMPO_DEBUG` environment variable:
   ```
   export TEMPO_DEBUG=true  # Enable debugging globally
   export TEMPO_DEBUG=false  # Disable debugging globally
   ```

2. **Per-module setting**: Set environment variables for specific modules:
   ```
   export TEMPO_DEBUG_TOKEN_GENERATOR=true  # Enable for token_generator only
   export TEMPO_DEBUG_MODEL_WRAPPER=false   # Disable for model_wrapper only
   ```

3. **Configuration file**: Edit the `MODULE_DEBUG_SETTINGS` dictionary in `src/utils/config.py`.

4. **Programmatic override**: When initializing a module, you can still explicitly set the debug mode:
   ```python
   # This overrides any configuration settings
   model_wrapper = ModelWrapper(model, debug_mode=True)
   ```

### Debug Modes by Default

In the central configuration, all modules default to the global `DEFAULT_DEBUG_MODE` setting, 
which is `True` by default. This ensures consistent debug behavior across all modules, while
still allowing for fine-grained control when needed.

### Logging

All modules use the `LoggingMixin` class from `src/utils/logging_utils.py` to provide consistent 
logging behavior:

```python
from src.utils.logging_utils import LoggingMixin

class MyClass(LoggingMixin):
    def __init__(self):
        super().__init__()
        self.setup_logging("my_class")
        
    def my_method(self):
        # Log messages are only written if debug_mode is enabled
        self.log("This is a log message")
        self.log("This is a warning", level="warning")
        self.log("This is an error", level="error")
```

All logs are written to the `logs/` directory in the project root.