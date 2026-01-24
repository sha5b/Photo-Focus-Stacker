# Context: Stack detection service for Photo Focus Stacker
# Purpose: Group image file paths into stacks using selectable strategies (auto, legacy, regex, fixed size).

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import natsort
except Exception:  # pragma: no cover
    natsort = None


StackItem = Tuple[str, List[str]]


_LEGACY_PATTERN = r"^(.*_\d+)-(\d+)$"


def detect_stacks(
    image_paths: Sequence[str],
    *,
    mode: str = "auto",
    fixed_stack_size: int = 0,
    regex_pattern: str = "",
) -> List[StackItem]:
    if not image_paths:
        return []

    normalized_paths = [os.path.normpath(p) for p in image_paths]

    if mode == "fixed_size":
        return _split_fixed_size(normalized_paths, fixed_stack_size)

    if mode == "regex":
        pattern = regex_pattern.strip()
        if not pattern:
            raise ValueError("regex_pattern is required when mode='regex'.")
        return _split_by_regex(normalized_paths, pattern)

    if mode == "legacy":
        return _split_by_regex(normalized_paths, _LEGACY_PATTERN)

    if mode == "common_suffix":
        # Heuristic mode: try to remove the last numeric run from filenames.
        return _split_by_common_suffix(normalized_paths)

    # auto
    auto = _split_by_regex(normalized_paths, _LEGACY_PATTERN)
    if _looks_reasonable(auto):
        return auto

    common = _split_by_common_suffix(normalized_paths)
    if _looks_reasonable(common):
        return common

    return [("stack", _natural_sort(normalized_paths))]


def _looks_reasonable(stacks: List[StackItem]) -> bool:
    if not stacks:
        return False
    # Consider it reasonable if we found multiple stacks, or one stack with multiple images.
    if len(stacks) > 1:
        return True
    return len(stacks[0][1]) >= 2


def _split_fixed_size(image_paths: Sequence[str], fixed_stack_size: int) -> List[StackItem]:
    if fixed_stack_size <= 0:
        raise ValueError("fixed_stack_size must be > 0 when mode='fixed_size'.")

    paths = _natural_sort(list(image_paths))
    stacks: List[StackItem] = []
    for idx in range(0, len(paths), fixed_stack_size):
        chunk = paths[idx : idx + fixed_stack_size]
        if not chunk:
            continue
        stacks.append((f"stack_{len(stacks) + 1}", chunk))
    return stacks


def _split_by_regex(image_paths: Sequence[str], pattern: str) -> List[StackItem]:
    compiled = re.compile(pattern, re.IGNORECASE)
    stacks: dict[str, List[str]] = {}

    for path in image_paths:
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)
        match = compiled.match(name)

        if match:
            base_name = str(match.group(1)).strip() or "default_stack"
        else:
            base_name = "default_stack"

        stacks.setdefault(base_name, []).append(path)

    stack_items: List[StackItem] = []
    for base_name, paths in stacks.items():
        sorted_paths = _natural_sort(paths)
        stack_items.append((base_name, sorted_paths))

    stack_items.sort(key=lambda item: item[1][0] if item[1] else "")
    return stack_items


_SUFFIX_NUM_PATTERN = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)$")


def _split_by_common_suffix(image_paths: Sequence[str]) -> List[StackItem]:
    stacks: dict[str, List[str]] = {}

    for path in image_paths:
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)

        # Remove separators at the end, then peel off the last numeric run.
        name_stripped = re.sub(r"[-_ ]+$", "", name)
        match = _SUFFIX_NUM_PATTERN.match(name_stripped)

        base_name = match.group("prefix") if match else name_stripped
        base_name = base_name.rstrip("-_ ")
        base_name = base_name if base_name else "default_stack"

        stacks.setdefault(base_name, []).append(path)

    stack_items: List[StackItem] = []
    for base_name, paths in stacks.items():
        stack_items.append((base_name, _natural_sort(paths)))

    stack_items.sort(key=lambda item: item[1][0] if item[1] else "")
    return stack_items


def _natural_sort(paths: Sequence[str]) -> List[str]:
    if natsort is not None:
        return list(natsort.natsorted(paths))
    return sorted(paths)
