"""
LLM client configuration for Graphiti MCP Server.
"""

import argparse
import logging
import os

from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from pydantic import BaseModel

from graphiti_mcp_server.config.settings import DEFAULT_LLM_MODEL, SMALL_LLM_MODEL

logger = logging.getLogger(__name__)


class GraphitiLLMConfig(BaseModel):
    """Configuration for the LLM client.

    Centralizes all LLM-specific configuration parameters including API keys and model selection.
    """

    api_key: str | None = None
    model: str = DEFAULT_LLM_MODEL
    small_model: str = SMALL_LLM_MODEL
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> 'GraphitiLLMConfig':
        """Create LLM configuration from environment variables."""
        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get('MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_LLM_MODEL

        # Get small_model from environment, or use default if not set or empty
        small_model_env = os.environ.get('SMALL_MODEL_NAME', '')
        small_model = small_model_env if small_model_env.strip() else SMALL_LLM_MODEL

        # Log if empty model was provided
        if model_env == '':
            logger.debug(
                f'MODEL_NAME environment variable not set, using default: {DEFAULT_LLM_MODEL}'
            )
        elif not model_env.strip():
            logger.warning(
                f'Empty MODEL_NAME environment variable, using default: {DEFAULT_LLM_MODEL}'
            )

        return cls(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model=model,
            small_model=small_model,
            temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiLLMConfig':
        """Create LLM configuration from CLI arguments, falling back to environment variables."""
        # Start with environment-based config
        config = cls.from_env()

        # CLI arguments override environment variables when provided
        # Use "is not None" to detect if argument was passed (even if empty string)
        if hasattr(args, 'model') and args.model is not None:
            if args.model.strip():
                config.model = args.model
            else:
                # Empty string explicitly provided - reset to default
                config.model = DEFAULT_LLM_MODEL
                logger.warning(f'Empty model name provided, using default: {DEFAULT_LLM_MODEL}')

        if hasattr(args, 'small_model') and args.small_model is not None:
            if args.small_model.strip():
                config.small_model = args.small_model
            else:
                # Empty string explicitly provided - reset to default
                config.small_model = SMALL_LLM_MODEL
                logger.warning(f'Empty small_model name provided, using default: {SMALL_LLM_MODEL}')

        if hasattr(args, 'temperature') and args.temperature is not None:
            config.temperature = args.temperature

        return config

    def create_client(self) -> LLMClient | None:
        """Create an LLM client based on this configuration.

        Returns:
            LLMClient instance, or None if api_key is not set.
            Callers should handle the None case appropriately.
        """
        if not self.api_key:
            logger.warning('OPENAI_API_KEY not set, LLM client will not be available')
            return None

        llm_client_config = LLMConfig(
            api_key=self.api_key, model=self.model, small_model=self.small_model
        )

        # Only set temperature if not using gpt-5, o1, or o3 models (they don't support temperature)
        if not any(x in self.model.lower() for x in ['gpt-5', 'o1', 'o3']):
            llm_client_config.temperature = self.temperature

        # Disable reasoning and verbosity parameters for gpt-5, o1, o3 models
        return OpenAIClient(config=llm_client_config, reasoning=None, verbosity=None)
