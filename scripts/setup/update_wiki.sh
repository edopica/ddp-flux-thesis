#!/bin/bash
# Script to update Wiki documentation locally.
# This uses your locally installed and patched version of Wiki.

echo "Updating Wiki documentation..."
# The -p flag prevents Ink from using raw mode/interactive TTY issues
openwiki -p --update

if [ $? -eq 0 ]; then
  echo "Wiki update completed successfully!"
  
  # Check if there are changes in the openwiki folder
  if [[ -n $(git status -s wiki/) ]]; then
    echo "Changes detected in wiki/. Staging them..."
    git add wiki/
    echo "Documentation changes staged for commit."
  else
    echo "No documentation changes were generated."
  fi
else
  echo "Error: Wiki update failed."
  exit 1
fi
