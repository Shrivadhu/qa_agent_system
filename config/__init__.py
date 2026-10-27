import os
import yaml
from pathlib import Path
from typing import Any, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Config:
    """Configuration manager for QA Agent system."""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self.load_config()
    
    def load_config(self, config_path: str = None):
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Replace environment variables
        self._replace_env_vars(self._config)
        # Apply overrides such as DATABASE__HOST=postgres -> database.host
        self._apply_env_overrides(self._config)
    
    def _replace_env_vars(self, config_dict: Dict):
        """Recursively replace ${VAR} with environment variables."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._replace_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config_dict[key] = os.getenv(env_var, value)
    
    def _apply_env_overrides(self, config_dict: Dict):
        """Override config values using SECTION__KEY environment variables."""
        for env_name, env_value in os.environ.items():
            if "__" not in env_name:
                continue
            parts = [p.lower() for p in env_name.split("__")]
            node = config_dict
            for part in parts[:-1]:
                node = node.get(part) if isinstance(node, dict) else None
                if node is None:
                    break
            if not isinstance(node, dict) or parts[-1] not in node:
                continue
            current = node[parts[-1]]
            if isinstance(current, bool):
                node[parts[-1]] = env_value.lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                try:
                    node[parts[-1]] = int(env_value)
                except ValueError:
                    pass
            elif isinstance(current, float):
                try:
                    node[parts[-1]] = float(env_value)
                except ValueError:
                    pass
            elif isinstance(current, (str, type(None))):
                node[parts[-1]] = env_value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example:
            config.get("database.host")
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    @property
    def database(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self._config.get('database', {})
    
    @property
    def models(self) -> Dict[str, Any]:
        """Get models configuration."""
        return self._config.get('models', {})
    
    @property
    def verdict_thresholds(self) -> Dict[str, float]:
        """Get verdict thresholds."""
        return self._config.get('verdict', {}).get('thresholds', {})

    @property
    def api_key(self) -> str:
        """Get API key from env or config."""
        return os.getenv("API_KEY", self.get("api.key", ""))

    @property
    def cors_origins(self) -> list:
        """Get CORS origins."""
        env_origins = os.getenv("CORS_ORIGINS")
        if env_origins:
            return [origin.strip() for origin in env_origins.split(",") if origin.strip()]
        return self.get("api.cors_origins", ["*"])


# Singleton instance
config = Config()
