"""
Base storage adapter interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from lib.utils import Config


class StorageAdapter(ABC):
    """
    Abstract base class for storage adapters.
    Defines the interface for storing and retrieving translation project data.
    """

    @abstractmethod
    async def load_config(self, project_id: str) -> Config:
        """Load project configuration"""
        pass

    @abstractmethod
    async def load_progress(self, project_id: str, language: str) -> Dict[str, str]:
        """Load translation progress for a specific language"""
        pass

    @abstractmethod
    async def save_progress(
        self, project_id: str, language: str, progress: Dict[str, str]
    ) -> None:
        """Save translation progress for a specific language"""
        pass

    @abstractmethod
    async def load_translations(self, project_id: str) -> List[Dict[str, str]]:
        """Load all translations"""
        pass

    @abstractmethod
    async def save_translations(
        self, project_id: str, translations: List[Dict[str, str]]
    ) -> None:
        """Save all translations"""
        pass

    @abstractmethod
    async def load_context(self, project_id: str, language: str) -> List[str]:
        """Load translation context"""
        pass

    @abstractmethod
    async def load_prompt(self, project_id: str, prompt_type: str) -> str:
        """Load translation prompt"""
        pass

    def set_active_language(self, language: Optional[str]) -> None:
        """
        Set the destination language for the current run.

        Adapters that support concurrent per-language runs use this to merge
        only the active language's column. No-op by default.
        """
        pass

    def set_overwrite_active_language(self, enabled: bool) -> None:
        """
        Allow overwriting non-empty cells for the active language (e.g.
        ``--regenerate``). No-op by default.
        """
        pass
