#!/bin/bash
# Script to update OpenWiki documentation locally.
# This uses your locally installed and patched version of OpenWiki.

echo "Updating OpenWiki documentation..."
# The -p flag prevents Ink from using raw mode/interactive TTY issues
openwiki -p --update

if [ $? -eq 0 ]; then
  echo "OpenWiki update completed successfully!"
  
  # Check if there are changes in the openwiki folder
  if [[ -n $(git status -s openwiki/) ]]; then
    echo "Changes detected in openwiki/. Staging them..."
    git add openwiki/
    echo "Documentation changes staged for commit."
  else
    echo "No documentation changes were generated."
  fi
else
  echo "Error: OpenWiki update failed."
  exit 1
fi
