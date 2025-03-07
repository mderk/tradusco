import os
from pathlib import Path
import re
from typing import Any, Optional

import aiofiles


class PromptManager:
    """Manages loading and handling of prompt templates."""

    def __init__(self, project_dir: Path):
        """Initialize the PromptManager.

        Args:
            project_dir: The project directory path
        """
        self.project_dir = project_dir
        self.prompts_dir = Path(__file__).parent.parent / "prompts"
        self._cache: dict[str, str] = {}
        self._required_vars: dict[str, set[str]] = {
            "translation": {"base_language", "dst_language", "phrases_json"},
            "json_fix": {"invalid_json"},
        }

    def _validate_prompt(
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
        if strict and prompt_type in self._required_vars:
            required = self._required_vars[prompt_type]
            # Use a regex to find all format variables in the template
            found_vars = set(re.findall(r"\{([^}]+)\}", template))
            missing = required - found_vars
            if missing:
                return False, f"Missing required variables: {', '.join(missing)}"

        return True, ""

    async def load_prompt(
        self,
        prompt_type: str,
        custom_prompt_path: Optional[str] = None,
        use_cache: bool = True,
        validate: bool = True,
        strict_validation: bool = False,
    ) -> str:
        """Load a prompt template, with optional custom override.

        Args:
            prompt_type: Type of prompt to load (e.g. 'translation', 'json_fix')
            custom_prompt_path: Optional path to a custom prompt file
            use_cache: Whether to use cached prompts
            validate: Whether to validate the prompt template
            strict_validation: Whether to enforce required variables

        Returns:
            The loaded prompt template string
        """
        # Check cache first if enabled
        if use_cache and prompt_type in self._cache:
            return self._cache[prompt_type]

        # Try custom prompt if provided
        if custom_prompt_path:
            try:
                if os.path.exists(custom_prompt_path):
                    async with aiofiles.open(
                        custom_prompt_path, "r", encoding="utf-8"
                    ) as f:
                        prompt = (await f.read()).strip()
                        if prompt:
                            if validate:
                                is_valid, error = self._validate_prompt(
                                    prompt_type, prompt, strict_validation
                                )
                                if not is_valid:
                                    print(
                                        f"Warning: Custom prompt validation failed: {error}"
                                    )
                                    if strict_validation:
                                        print("Falling back to default prompt.")
                                        return await self._load_default_prompt(
                                            prompt_type,
                                            use_cache,
                                            validate,
                                            strict_validation,
                                        )
                            if use_cache:
                                self._cache[prompt_type] = prompt
                            return prompt
                else:
                    print(
                        f"Warning: Custom prompt file {custom_prompt_path} not found."
                    )
                    return ""  # Return empty string for nonexistent custom prompt
            except Exception as e:
                print(
                    f"Warning: Error reading custom prompt file {custom_prompt_path}: {e}"
                )
                return ""  # Return empty string on error

        # Fall back to default prompt only if no custom prompt was specified
        if custom_prompt_path is None:
            return await self._load_default_prompt(
                prompt_type, use_cache, validate, strict_validation
            )
        return ""

    async def _load_default_prompt(
        self, prompt_type: str, use_cache: bool, validate: bool, strict_validation: bool
    ) -> str:
        """Load a default prompt from the prompts directory.

        Args:
            prompt_type: Type of prompt to load
            use_cache: Whether to use cached prompts
            validate: Whether to validate the prompt
            strict_validation: Whether to enforce required variables

        Returns:
            The loaded prompt template string
        """
        # Support prompt templates in subdirectories
        prompt_paths = [
            self.prompts_dir / f"{prompt_type}.txt",
            self.prompts_dir / prompt_type / "prompt.txt",
        ]

        for path in prompt_paths:
            try:
                if path.exists():
                    async with aiofiles.open(path, "r", encoding="utf-8") as f:
                        prompt = (await f.read()).strip()
                        if prompt:
                            if validate:
                                is_valid, error = self._validate_prompt(
                                    prompt_type, prompt, strict_validation
                                )
                                if not is_valid:
                                    print(
                                        f"Warning: Default prompt validation failed: {error}"
                                    )
                                    if strict_validation:
                                        continue
                            if use_cache:
                                self._cache[prompt_type] = prompt
                            return prompt
            except Exception as e:
                print(f"Warning: Error reading default prompt file {path}: {e}")
                continue

        print(f"Warning: No valid prompt found for type '{prompt_type}'")
        return ""

    def format_prompt(self, template: str, **kwargs: Any) -> str:
        """Format a prompt template with the provided variables.

        Args:
            template: The prompt template string
            **kwargs: Variables to format into the template

        Returns:
            The formatted prompt string
        """
        try:
            # First check if all required variables are provided
            required_vars = set(re.findall(r"\{([^}]+)\}", template))
            missing = required_vars - set(kwargs.keys())
            if missing:
                print(
                    f"Warning: Missing variables in format_prompt: {', '.join(missing)}"
                )
                return template

            return template.format(**kwargs)
        except KeyError as e:
            print(f"Warning: Missing required variable in prompt template: {e}")
            return template
        except ValueError as e:
            print(f"Warning: Invalid format in prompt template: {e}")
            return template
        except Exception as e:
            print(f"Warning: Error formatting prompt template: {e}")
            return template

    def clear_cache(self, prompt_type: Optional[str] = None) -> None:
        """Clear the prompt cache.

        Args:
            prompt_type: Optional specific prompt type to clear from cache
        """
        if prompt_type:
            self._cache.pop(prompt_type, None)
        else:
            self._cache.clear()
