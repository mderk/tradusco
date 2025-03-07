import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from lib.TranslationProject import TranslationProject


class TestContextHandling(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        """Set up test environment before each test"""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.project_name = "test_project"
        self.project_dir = Path(self.temp_dir) / "projects" / self.project_name

        # Create project directory structure
        os.makedirs(self.project_dir, exist_ok=True)

        # Create a basic config.json
        self.config = {
            "name": self.project_name,
            "sourceFile": "translations.csv",
            "languages": ["en", "es"],
            "baseLanguage": "en",
            "keyColumn": "en",
        }
        with open(self.project_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f)

        # Create a basic translations.csv with context column
        self.csv_content = (
            "en,es,context\n"
            "hello,hola,formal greeting\n"
            "goodbye,,casual farewell\n"
            "welcome,,\n"
        )
        with open(self.project_dir / "translations.csv", "w", encoding="utf-8") as f:
            f.write(self.csv_content)

        # Create mock for Path that returns our temp directory for projects
        def path_side_effect(path_str):
            if path_str == "projects":
                return Path(self.temp_dir) / "projects"
            if isinstance(path_str, Path):
                return path_str
            # Handle project directory paths
            if str(path_str).startswith("projects/"):
                return Path(self.temp_dir) / path_str
            return Path(path_str)

        self.patcher = patch("lib.TranslationProject.Path")
        self.mock_path = self.patcher.start()
        self.mock_path.side_effect = path_side_effect
        # Make sure Path() calls also work
        self.mock_path.return_value = Path()

    async def asyncTearDown(self):
        """Clean up after each test"""
        self.patcher.stop()
        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir)

    async def create_test_project(self, context=None, context_file=None):
        """Helper method to create a test project instance"""
        return await TranslationProject.create(
            self.project_name, "es", context=context, context_file=context_file
        )

    async def test_load_global_context_from_string(self):
        """Test loading context provided directly as a string"""
        test_context = "This is a test context"
        project = await self.create_test_project(context=test_context)
        loaded_context = await project._load_context()
        self.assertEqual(loaded_context, test_context)

    async def test_load_global_context_from_file(self):
        """Test loading context from a context file"""
        # Create a test context file
        context_file = Path(self.temp_dir) / "test_context.txt"
        test_context = "This is a test context from file"
        with open(context_file, "w", encoding="utf-8") as f:
            f.write(test_context)

        project = await self.create_test_project(context_file=str(context_file))
        loaded_context = await project._load_context()
        self.assertEqual(loaded_context, test_context)

    async def test_load_project_context_file(self):
        """Test loading context from project's context.md/txt file"""
        # Create both context files to test priority
        test_context_md = "Context from MD file"
        test_context_txt = "Context from TXT file"

        with open(self.project_dir / "context.md", "w", encoding="utf-8") as f:
            f.write(test_context_md)
        with open(self.project_dir / "context.txt", "w", encoding="utf-8") as f:
            f.write(test_context_txt)

        project = await self.create_test_project()
        loaded_context = await project._load_context()

        # Both contexts should be loaded and combined
        self.assertIn(test_context_md, loaded_context)
        self.assertIn(test_context_txt, loaded_context)

    async def test_combine_multiple_contexts(self):
        """Test combining contexts from multiple sources"""
        # Create project context file
        project_context = "Project context"
        with open(self.project_dir / "context.md", "w", encoding="utf-8") as f:
            f.write(project_context)

        # Create external context file
        context_file = Path(self.temp_dir) / "external_context.txt"
        external_context = "External context"
        with open(context_file, "w", encoding="utf-8") as f:
            f.write(external_context)

        # Direct context string
        direct_context = "Direct context"

        project = await self.create_test_project(
            context=direct_context, context_file=str(context_file)
        )
        loaded_context = await project._load_context()

        # All contexts should be present
        self.assertIn(project_context, loaded_context)
        self.assertIn(external_context, loaded_context)
        self.assertIn(direct_context, loaded_context)

    async def test_phrase_specific_context_loading(self):
        """Test loading and formatting phrase-specific contexts"""
        project = await self.create_test_project()
        translations = await project._load_translations()

        # Test creating batch prompt with phrase-specific contexts
        phrases = ["hello", "goodbye"]
        indices = [0, 1]  # Indices of the phrases in translations list

        prompt = await project._create_batch_prompt(phrases, translations, indices)

        # Verify that phrase contexts are included in the prompt
        self.assertIn('"hello": "formal greeting"', prompt)
        self.assertIn('"goodbye": "casual farewell"', prompt)

    async def test_empty_contexts(self):
        """Test behavior with empty contexts"""
        project = await self.create_test_project()
        translations = await project._load_translations()

        # Test with phrases that have no context
        phrases = ["welcome"]
        indices = [2]  # Index of 'welcome' which has no context

        prompt = await project._create_batch_prompt(phrases, translations, indices)

        # Verify that empty contexts don't create issues
        self.assertNotIn('"welcome":', prompt)  # Should not include empty contexts
        self.assertIn('"welcome"', prompt)  # But should include the phrase

    async def test_context_in_prompt_format(self):
        """Test that contexts are properly formatted in the prompt"""
        test_context = "Global test context"
        project = await self.create_test_project(context=test_context)
        translations = await project._load_translations()

        phrases = ["hello", "welcome"]
        indices = [0, 2]

        prompt = await project._create_batch_prompt(phrases, translations, indices)

        # Check global context formatting
        self.assertIn("Global Translation Context:", prompt)
        self.assertIn(test_context, prompt)

        # Check phrase-specific context formatting
        self.assertIn("Phrase-specific contexts:", prompt)
        self.assertIn('"hello": "formal greeting"', prompt)

    async def test_missing_context_file(self):
        """Test graceful handling of missing context files"""
        # Try to load from a non-existent context file
        non_existent_file = str(Path(self.temp_dir) / "non_existent.txt")
        project = await self.create_test_project(context_file=non_existent_file)

        # Should not raise an exception
        loaded_context = await project._load_context()
        self.assertEqual(loaded_context, "")

    async def test_invalid_context_in_csv(self):
        """Test handling of invalid contexts in CSV file"""
        # Create CSV with invalid context format
        invalid_csv = "en,es,context\n" "test1,,{invalid_json}\n" "test2,,\n"

        with open(self.project_dir / "translations.csv", "w", encoding="utf-8") as f:
            f.write(invalid_csv)

        project = await self.create_test_project()
        translations = await project._load_translations()

        # Should not raise an exception when creating prompt
        phrases = ["test1"]
        indices = [0]
        prompt = await project._create_batch_prompt(phrases, translations, indices)

        # The invalid context should be included as is
        self.assertIn('"test1": "{invalid_json}"', prompt)
