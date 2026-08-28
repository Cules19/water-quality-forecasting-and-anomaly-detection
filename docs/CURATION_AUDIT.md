# Repository curation audit

Initial curation: 2026-08-25<br>
Final validation: 2026-08-28

## Resulting code tree

- 240 tracked files in the final tree.
- 133 Python files.
- 48 retained JSON experiment configurations.
- 46 Markdown Preview documents, including stage READMEs and scientific documentation.
- No tracked file exceeds 1 MB; datasets, checkpoints, predictions, and caches remain outside Git.

## Validation

- All Python source files parse successfully under Python 3.12.
- All 48 JSON configurations parse successfully.
- All local Markdown links resolve.
- No machine-specific user-home path remains in active code or documentation.
- The user-provided USGS API key is absent from the proposed tree.
- The complete pytest suite ran 79 tests successfully.
- `git diff --cached --check` passes.

## Source and preservation boundary

The teacher/reference repository was read only. The curated snapshot copies selected reusable modules, including the observed-target split-alignment correction present in the working tree, without committing to or otherwise changing the teacher repository.

The final repository was prepared and validated in a separate preview worktree. Its publication target is the user's `Cules19/git` repository only. The pre-cleanup remote state is preserved by a local backup branch.

Large local evidence is preserved under `Pollution_ID_USB_ARCHIVE_2026-08-25`; final source/destination file counts and total bytes match for all 97 selected evidence directories.
