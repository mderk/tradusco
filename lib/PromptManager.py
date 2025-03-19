import os
import json
from pathlib import Path
import re
from typing import Optional

import aiofiles
from pydantic import BaseModel

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
        self.prompts_dir = Path(__file__).parent.parent / "prompts"
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

    async def _load_prompt_from_path(
        self,
        prompt_type: str,
        path: Path | str,
        is_default: bool = False,
    ) -> str:
        """Load a prompt from a specific path with filesystem operations.

        Args:
            prompt_type: Type of prompt to load
            path: Path to the prompt file
            is_default: Whether this is loading a default prompt

        Returns:
            The loaded prompt template string
        """
        try:
            if os.path.exists(path):
                async with aiofiles.open(path, "r", encoding="utf-8") as f:
                    prompt = (await f.read()).strip()
                    if prompt:
                        return prompt
            else:
                source = "Default" if is_default else "Custom"
                print(f"Warning: {source} {prompt_type} prompt file {path} not found.")
        except Exception as e:
            source = "Default" if is_default else "Custom"
            print(
                f"Warning: Error reading {source} {prompt_type} prompt file {path}: {e}"
            )
        return ""

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

        # Fallback to default prompts in the package if nothing found in storage
        prompt_paths = [
            self.prompts_dir / f"{prompt_type}.txt",
            self.prompts_dir / prompt_type / "prompt.txt",
        ]

        for path in prompt_paths:
            prompt = await self._load_prompt_from_path(
                prompt_type,
                path,
                is_default=True,
            )
            if prompt:
                if validate:
                    is_valid, error = self.validate_prompt(
                        prompt_type, prompt, strict_validation
                    )
                    if error:
                        source = "Default"
                        print(
                            f"{'Warning' if is_valid else 'Error'}: {source} prompt validation failed: {error}"
                        )

                    if not is_valid and strict_validation:
                        continue
                    elif use_cache:
                        # Cache prompt if not in strict mode or if valid
                        self._cache[prompt_type] = prompt
                elif use_cache:
                    # If not validating, only cache if requested
                    self._cache[prompt_type] = prompt
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
