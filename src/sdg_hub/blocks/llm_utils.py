# SPDX-License-Identifier: Apache-2.0
"""LLM chat completion utilities for input formatting and output parsing.

This module provides blocks for handling LLM chat completions, including:
- StringParserBlock: Parse and post-process LLM outputs
"""

# Standard
from typing import List, Optional, Union
import re

# Third Party
from datasets import Dataset

# Local
from ..logger_config import setup_logger
from ..registry import BlockRegistry
from .block import Block

logger = setup_logger(__name__)


@BlockRegistry.register("StringParserBlock")
class StringParserBlock(Block):
    """Block for parsing and post-processing LLM outputs.

    This block handles output parsing using start/end tags, custom regex patterns,
    and cleanup operations. It duplicates the parsing functionality from LLMBlock.

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name(s) containing raw LLM output.
    output_cols : Union[str, List[str]]
        Output column name(s) for parsed results.
    start_tags : List[str], optional
        List of start tags for tag-based parsing, by default [].
    end_tags : List[str], optional
        List of end tags for tag-based parsing, by default [].
    parsing_pattern : Optional[str], optional
        Regex pattern for custom parsing, by default None.
    parser_cleanup_tags : Optional[List[str]], optional
        List of tags to clean from parsed output, by default None.
    """

    def __init__(
        self,
        block_name: str,
        input_cols: Union[str, List[str]],
        output_cols: Union[str, List[str]],
        start_tags: List[str] = [],
        end_tags: List[str] = [],
        parsing_pattern: Optional[str] = None,
        parser_cleanup_tags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(block_name)
        self.input_cols = [input_cols] if isinstance(input_cols, str) else input_cols
        self.output_cols = (
            [output_cols] if isinstance(output_cols, str) else output_cols
        )
        self.start_tags = start_tags
        self.end_tags = end_tags
        self.parsing_pattern = parsing_pattern
        self.parser_cleanup_tags = parser_cleanup_tags

        # Validate the block configuration
        if len(self.input_cols) == 0:
            raise ValueError("StringParserBlock expects at least one input column")
        elif len(self.input_cols) > 1:
            logger.warning(
                f"StringParserBlock expects exactly one input column, but got {len(self.input_cols)}. "
                f"Using the first column: {self.input_cols[0]}"
            )

    def _extract_matches(
        self, text: str, start_tag: Optional[str], end_tag: Optional[str]
    ) -> List[str]:
        if not text:
            return []
        if not start_tag and not end_tag:
            return [text.strip()]

        pattern = ""
        if start_tag:
            pattern += re.escape(start_tag)
        pattern += r"(.*?)"
        if end_tag:
            pattern += re.escape(end_tag)
        elif start_tag:
            pattern += "$"

        return [match.strip() for match in re.findall(pattern, text, re.DOTALL)]

    def _parse(self, generated_string: str) -> dict[str, list[str]]:
        matches: dict[str, list[str]] = {}

        if self.parsing_pattern is not None:
            pattern = re.compile(self.parsing_pattern, re.DOTALL)
            all_matches = pattern.findall(generated_string)
            matches = {column_name: [] for column_name in self.output_cols}
            if all_matches and isinstance(all_matches[0], tuple):
                for match in all_matches:
                    for column_name, value in zip(self.output_cols, match):
                        value = value.strip()
                        if self.parser_cleanup_tags:
                            for clean_tag in self.parser_cleanup_tags:
                                value = value.replace(clean_tag, "")
                        matches[column_name].append(value)
            else:
                cleaned_matches = []
                for match in all_matches:
                    value = match.strip()
                    if self.parser_cleanup_tags:
                        for clean_tag in self.parser_cleanup_tags:
                            value = value.replace(clean_tag, "")
                    cleaned_matches.append(value)
                matches[self.output_cols[0]] = cleaned_matches if all_matches else []
        else:
            # Initialize all output columns with empty lists
            matches = {column_name: [] for column_name in self.output_cols}

            # Process tag pairs for matching columns
            for start_tag, end_tag, output_col in zip(
                self.start_tags,
                self.end_tags,
                self.output_cols,
            ):
                matches[output_col] = self._extract_matches(
                    generated_string, start_tag, end_tag
                )
        return matches

    def _generate(self, sample: dict) -> List[dict]:
        input_column = self.input_cols[0]
        if input_column not in sample:
            logger.warning(
                f"Input column '{input_column}' not found in sample: {sample}"
            )
            return []

        raw_output = sample[input_column]
        parsed_outputs = self._parse(raw_output)

        if not parsed_outputs or not any(
            len(value) > 0 for value in parsed_outputs.values()
        ):
            return []

        result = []
        max_length = max(len(value) for value in parsed_outputs.values())
        for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
            result.append({**sample, **dict(zip(parsed_outputs.keys(), values))})
        return result

    def generate(self, samples: Dataset) -> Dataset:
        logger.debug("Parsing outputs for {} samples".format(len(samples)))
        if len(samples) == 0:
            logger.warning("No samples to parse, returning empty dataset")
            return Dataset.from_list([])

        new_data = []
        for sample in samples:
            new_data.extend(self._generate(sample))
        return Dataset.from_list(new_data)
