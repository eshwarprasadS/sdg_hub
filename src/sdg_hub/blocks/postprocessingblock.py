# SPDX-License-Identifier: Apache-2.0
"""Post-processing blocks for parsing and cleaning LLM outputs.

This module provides blocks for parsing and post-processing language model outputs.
"""

# Standard
from typing import Any, Dict, List, Optional, Union
import re

# Third Party
from datasets import Dataset

# Local
from .block import Block
from ..logger_config import setup_logger
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register("PostProcessingBlock")
class PostProcessingBlock(Block):
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
    parser_name : Optional[str], optional
        Name of the parser to use ("custom" for regex), by default None.
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
        parser_name: Optional[str] = None,
        parsing_pattern: Optional[str] = None,
        parser_cleanup_tags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(block_name)
        self.input_cols = [input_cols] if isinstance(input_cols, str) else input_cols
        self.output_cols = [output_cols] if isinstance(output_cols, str) else output_cols
        self.start_tags = start_tags
        self.end_tags = end_tags
        self.parser_name = parser_name
        self.parsing_pattern = parsing_pattern
        self.parser_cleanup_tags = parser_cleanup_tags

        # For this block, we expect exactly one input column and one or more output columns
        if len(self.input_cols) == 0:
            raise ValueError("PostProcessingBlock expects at least one input column")
        elif len(self.input_cols) > 1:
            logger.warning(
                f"PostProcessingBlock expects exactly one input column, but got {len(self.input_cols)}. "
                f"Using the first column: {self.input_cols[0]}"
            )

    def _extract_matches(
        self, text: str, start_tag: Optional[str], end_tag: Optional[str]
    ) -> List[str]:
        """Extract matches from text using start and end tags.

        Parameters
        ----------
        text : str
            Text to extract matches from.
        start_tag : Optional[str]
            Start tag to match.
        end_tag : Optional[str]
            End tag to match.

        Returns
        -------
        List[str]
            List of extracted matches.
        """
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
            # Enforce matching till end of string when only start_tag is provided.
            pattern += "$"

        return [match.strip() for match in re.findall(pattern, text, re.DOTALL)]

    def _parse(self, generated_string: str) -> dict:
        """Parse the generated string into structured output.

        Parameters
        ----------
        generated_string : str
            Raw generated string from LLM.

        Returns
        -------
        dict
            Parsed output with column names as keys.
        """
        matches = {}

        if self.parser_name is not None and self.parser_name == "custom":
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
                matches[self.output_cols[0]] = (
                    [match.strip() for match in all_matches] if all_matches else []
                )
        else:
            # Initialize all output columns with empty lists
            matches = {column_name: [] for column_name in self.output_cols}
            
            for start_tag, end_tag, output_col in zip(
                self.start_tags,
                self.end_tags,
                self.output_cols,
            ):
                matches[output_col] = self._extract_matches(
                    generated_string, start_tag, end_tag
                )

        return matches

    def generate(self, samples: Dataset, **gen_kwargs: Dict[str, Any]) -> Dataset:
        """Generate parsed output from raw LLM outputs.

        Parameters
        ----------
        samples : Dataset
            Input dataset containing raw LLM outputs.
        **gen_kwargs : Dict[str, Any]
            Additional keyword arguments (not used in this block).

        Returns
        -------
        Dataset
            Dataset with parsed outputs added as new columns.
        """
        logger.debug("Parsing outputs for {} samples".format(len(samples)))

        if len(samples) == 0:
            logger.warning("No samples to parse, returning empty dataset")
            return Dataset.from_list([])

        input_column = self.input_cols[0]
        new_data = []
        for sample in samples:
            if input_column not in sample:
                logger.warning(
                    f"Input column '{input_column}' not found in sample: {sample}"
                )
                continue

            raw_output = sample[input_column]
            parsed_outputs = self._parse(raw_output)
            
            max_length = max(len(value) for value in parsed_outputs.values())
            for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
                new_data.append({**sample, **dict(zip(parsed_outputs.keys(), values))})

        return Dataset.from_list(new_data) 