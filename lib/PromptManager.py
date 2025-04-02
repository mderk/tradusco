import os
import json
from pathlib import Path
import re
from typing import Optional

from pydantic import BaseModel

import prompts
from lib.storage.base import StorageAdapter

DEBUG = os.environ.get("TRADUSCO_DEBUG")


class PromptManager:
    """Manages loading and handling of prompt templates."""

    def __init__(self, storage: StorageAdapter, project_id: str):
        """Initialize the PromptManager.

        Args:
            storage: The storage adapter to use for loading prompts
            project_id: The project identifier
        """
        self.storage = storage
        self.project_id = project_id
        self._cache: dict[str, str] = {}
        self._required_vars: dict[str, set[str]] = {
            "translation": {"base_language", "dst_language", "phrases_json"},
        }

    def validate_prompt(
        self, prompt_type: str, template: str, strict: bool = False
    ) -> tuple[bool, str]:
        """Validate a prompt template.

        Args:
            prompt_type: Type of prompt being validated
            template: The prompt template string
            strict: Whether to enforce required variables (only used for actual prompts)

        Returns:
            A tuple of (is_valid, error_message)
        """
        if not template:
            return False, "Empty prompt template"

        # Check for required variables only in strict mode
        if prompt_type in self._required_vars:
            required = self._required_vars[prompt_type]
            # Use a regex to find all format variables in the template
            found_vars = set(re.findall(r"\{([^}]+)\}", template))
            missing = required - found_vars
            if missing:
                error = f"Missing required variables: {', '.join(missing)}"
                return (not strict), error

        return True, ""

    def get_default_prompt(self, prompt_type: str) -> str:
        """Get the default prompt for a given prompt type.

        Args:
            prompt_type: Type of prompt to get

        Returns:
            The default prompt string
        """
        return getattr(prompts, prompt_type)

    async def load_prompt(
        self,
        prompt_type: str,
        use_cache: bool = True,
        validate: bool = True,
        strict_validation: bool = False,
    ) -> str:
        """Load a prompt template from the storage adapter.

        Args:
            prompt_type: Type of prompt to load (e.g. 'translation', 'json_fix')
            use_cache: Whether to use cached prompts
            validate: Whether to validate the prompt template
            strict_validation: Whether to enforce required variables

        Returns:
            The loaded prompt template string
        """
        # Check cache first if enabled
        if use_cache and prompt_type in self._cache:
            return self._cache[prompt_type]

        # Load prompt from storage adapter
        try:
            prompt = await self.storage.load_prompt(self.project_id, prompt_type)

            if prompt:
                if validate:
                    is_valid, error = self.validate_prompt(
                        prompt_type, prompt, strict_validation
                    )
                    if error:
                        print(
                            f"{'Warning' if is_valid else 'Error'}: Prompt validation failed: {error}"
                        )

                    if not is_valid and strict_validation:
                        return ""
                    elif use_cache:
                        # Cache prompt if not in strict mode or if valid
                        self._cache[prompt_type] = prompt
                elif use_cache:
                    # If not validating, only cache if requested
                    self._cache[prompt_type] = prompt

                return prompt
        except Exception as e:
            print(f"Warning: Error loading prompt '{prompt_type}': {e}")

        prompt = self.get_default_prompt(prompt_type)
        if prompt:
            return prompt

        print(f"Warning: No valid prompt found for type '{prompt_type}'")
        return ""

    def format_prompt(self, template: str, data: BaseModel) -> str | None:
        """Format a prompt template with the provided variables.

        Args:
            template: The prompt template string
            data: template data

        Returns:
            The formatted prompt string
        """
        try:
            data_dump = data.model_dump()
            # First check if all required variables are provided
            required_vars = set(re.findall(r"\{([^}]+)\}", template))
            for key in list(data_dump.keys()):
                if f"{key}_json" in required_vars:
                    data_dump[f"{key}_json"] = json.dumps(data_dump[key])

            return template.format(**data_dump)
        except KeyError as e:
            print(f"Warning: Missing required variable in prompt template: {e}")
            if DEBUG:
                raise e
            return None
        except ValueError as e:
            print(f"Warning: Invalid format in prompt template: {e}")
            if DEBUG:
                raise e
            return None
        except Exception as e:
            print(f"Warning: Error formatting prompt template: {e}")
            if DEBUG:
                raise e
            return None

    def clear_cache(self, prompt_type: Optional[str] = None) -> None:
        """Clear the prompt cache.

        Args:
            prompt_type: Optional specific prompt type to clear from cache
        """
        if prompt_type:
            self._cache.pop(prompt_type, None)
        else:
            self._cache.clear()
