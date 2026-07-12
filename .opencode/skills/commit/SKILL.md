---
name: commit
description: Use ONLY when the user asks to commit code, save progress, or make a git commit.
---

# Commit Workflow

Before creating any git commit in this repository, you MUST follow these exact steps:

1. **Update Wiki**: Execute `./scripts/remote/update_wiki.sh` in the root directory. (This script automatically updates the `openwiki/` documentation and stages any changes).
2. **Review Staged Changes**: Run `git status` and `git diff --cached` to review what is about to be committed. If no files are staged, examine `git status` and ask the user what to stage.
3. **Draft and Commit**: Write a concise, descriptive commit message based on the staged changes and execute `git commit -m "<message>"`.
