"""
Configuration management for the Vector-Enhanced Temporal Knowledge Graph System.

This module handles loading and validation of configuration from YAML files
and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

from .models import DomainType, EmbeddingProvider


class EmbeddingProviderConfig(BaseModel):
    """Configuration for a specific embedding provider."""
    
    model: str
    api_key_env: Optional[str] = None
    device: str = "cpu"
    timeout_seconds: int = 30
    max_retries: int = 3


class EmbeddingConfig(BaseModel):
    """Embedding configuration with provider settings."""
    
    primary_provider: EmbeddingProvider = EmbeddingProvider.GOOGLE
    fallback_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    local_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    batch_size: int = Field(default=32, ge=1, le=128)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    
    providers: Dict[str, EmbeddingProviderConfig] = Field(default_factory=dict)
    
    @validator('providers')
    def validate_providers(cls, v):
        """Ensure required providers are configured."""
        required_providers = ['google', 'openai', 'sentence_transformers']
        for provider in required_providers:
            if provider not in v:
                # Set default configuration
                if provider == 'google':
                    v[provider] = EmbeddingProviderConfig(
                        model="text-embedding-004",
                        api_key_env="GOOGLE_API_KEY"
                    )
                elif provider == 'openai':
                    v[provider] = EmbeddingProviderConfig(
                        model="text-embedding-3-small", 
                        api_key_env="OPENAI_API_KEY"
                    )
                elif provider == 'sentence_transformers':
                    v[provider] = EmbeddingProviderConfig(
                        model="all-MiniLM-L6-v2",
                        device="cpu"
                    )
        return v


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    
    provider: str = "chromadb"
    persist_directory: str = "./data/vector_store"
    collections: List[Dict[str, str]] = Field(default_factory=list)
    
    @validator('collections')
    def ensure_default_collections(cls, v):
        """Ensure default collections are configured."""
        default_collections = [
            {"name": "document_chunks", "embedding_function": "google"},
            {"name": "atomic_facts", "embedding_function": "google"},
            {"name": "temporal_events", "embedding_function": "google"},
            {"name": "domain_entities", "embedding_function": "google"}
        ]
        
        if not v:
            return default_collections
        
        # Check if all default collections exist
        existing_names = {col.get("name") for col in v}
        for default_col in default_collections:
            if default_col["name"] not in existing_names:
                v.append(default_col)
        
        return v


class SearchConfig(BaseModel):
    """Search configuration parameters."""
    
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=50, ge=1, le=1000)
    
    hybrid_weights: Dict[str, float] = Field(default_factory=lambda: {
        "vector": 0.6,
        "graph": 0.3,
        "lexical": 0.1
    })
    
    ranking: Dict[str, float] = Field(default_factory=lambda: {
        "temporal_weight": 0.2,
        "confidence_weight": 0.3,
        "diversity_factor": 0.1
    })
    
    @validator('hybrid_weights')
    def validate_hybrid_weights(cls, v):
        """Ensure hybrid weights sum to 1.0."""
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Hybrid weights must sum to 1.0, got {total}")
        return v


class TemporalConfig(BaseModel):
    """Temporal search configuration."""
    
    enable_temporal_search: bool = True
    temporal_decay_factor: float = Field(default=0.95, ge=0.0, le=1.0)
    max_temporal_range_days: int = Field(default=365, ge=1)
    temporal_resolution: str = "day"


class GraphConfig(BaseModel):
    """Knowledge graph configuration."""
    
    backend: str = "networkx"
    max_depth: int = Field(default=3, ge=1, le=10)
    relationship_weights: bool = True


class KnowledgeDomainConfig(BaseModel):
    """Configuration for a knowledge domain."""
    
    name: str
    confidence_threshold: float = Field(ge=0.0, le=1.0)


class APIConfig(BaseModel):
    """API server configuration."""
    
    host: str = "localhost"
    port: int = Field(default=8000, ge=1024, le=65535)
    workers: int = Field(default=1, ge=1)
    reload: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = "./logs/app.log"


class PerformanceConfig(BaseModel):
    """Performance and optimization configuration."""
    
    vector_cache_size: int = Field(default=1000, ge=0)
    query_timeout_seconds: int = Field(default=30, ge=1)
    max_concurrent_queries: int = Field(default=50, ge=1)
    enable_metrics: bool = True


class Config(BaseModel):
    """Main configuration class."""
    
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    knowledge_domains: List[KnowledgeDomainConfig] = Field(default_factory=list)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    @validator('knowledge_domains')
    def ensure_default_domains(cls, v):
        """Ensure default knowledge domains are configured."""
        if not v:
            return [
                KnowledgeDomainConfig(name="general", confidence_threshold=0.8),
                KnowledgeDomainConfig(name="technical", confidence_threshold=0.9),
                KnowledgeDomainConfig(name="business", confidence_threshold=0.85)
            ]
        return v


class ConfigManager:
    """Configuration manager for loading and managing configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        # Load environment variables
        load_dotenv()
        
        # Determine config path
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "./config/config.yaml")
        
        self.config_path = Path(config_path)
        self._config: Optional[Config] = None
    
    def load_config(self) -> Config:
        """Load configuration from file."""
        if self._config is not None:
            return self._config
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Replace environment variable references
                config_data = self._resolve_env_vars(config_data)
                
                self._config = Config(**config_data)
            else:
                # Use default configuration
                self._config = Config()
                
                # Create default config file
                self._save_default_config()
                
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {self.config_path}: {e}")
        
        return self._config
    
    def _resolve_env_vars(self, data: Any) -> Any:
        """Recursively resolve environment variable references in config data."""
        if isinstance(data, dict):
            return {key: self._resolve_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._resolve_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            # Extract environment variable name
            env_var = data[2:-1]
            default_value = None
            
            # Handle default values: ${VAR_NAME:default_value}
            if ":" in env_var:
                env_var, default_value = env_var.split(":", 1)
            
            return os.getenv(env_var, default_value)
        else:
            return data
    
    def _save_default_config(self):
        """Save default configuration to file."""
        try:
            # Create config directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate default config YAML
            default_config = self._config.dict()
            
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save default config to {self.config_path}: {e}")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider from environment variables."""
        if not self._config:
            self.load_config()
        
        provider_config = self._config.embedding.providers.get(provider)
        if provider_config and provider_config.api_key_env:
            return os.getenv(provider_config.api_key_env)
        
        return None
    
    def reload_config(self) -> Config:
        """Reload configuration from file."""
        self._config = None
        return self.load_config()
    
    @property
    def config(self) -> Config:
        """Get current configuration."""
        if self._config is None:
            self.load_config()
        return self._config


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> Config:
    """Get the current configuration."""
    return config_manager.config


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider."""
    return config_manager.get_api_key(provider)