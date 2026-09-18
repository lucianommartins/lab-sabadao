#!/usr/bin/env bash
# Pre-publish leak gate: fail if any tracked file contains an internal-identifier or secret
# pattern. Runs in CI (.github/workflows/leak-gate.yml) and as a pre-commit hook
# (.pre-commit-config.yaml).
#
# Design note: this file SHIPS publicly, so it must not itself contain the sensitive product
# or serving codenames it guards - a denylist that names the secrets re-leaks them. Only
# GENERIC, safe-to-ship patterns live here (corporate emails, common secret prefixes, private
# key headers, cloud-metadata hosts). Supply the PROJECT-SPECIFIC codename denylist privately
# via the GBENCH_LEAK_PATTERNS env var (a CI repo variable/secret, or a local shell export);
# it is OR-ed into the scan when set. Example:
#   GBENCH_LEAK_PATTERNS='codenameA|codenameB|/abs/dev/path' bash scripts/check_no_leaks.sh
set -uo pipefail

# Generic, safe-to-ship denylist: corporate author emails, private-key headers, and common
# cloud/API secret prefixes. (Legitimate infrastructure hostnames such as the GCP metadata
# server are NOT here - they are real integration code, gated at runtime, not a leak.)
generic='@google\.com|-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|ghp_[0-9A-Za-z]{36}|sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{35}'

pattern="$generic"
if [ -n "${GBENCH_LEAK_PATTERNS:-}" ]; then
  pattern="$generic|${GBENCH_LEAK_PATTERNS}"
fi

# Tracked files only. Exclude this script (it names the generic patterns by construction) and
# the plugins symlink (an optional private tree that is not part of the public package).
hits=$(git grep -nIE "$pattern" -- \
  ':(exclude)scripts/check_no_leaks.sh' \
  ':(exclude)gbench/plugins' 2>/dev/null || true)

if [ -n "$hits" ]; then
  echo "❌ leak-gate: internal-identifier / secret pattern found in tracked files:" >&2
  echo "$hits" >&2
  echo "" >&2
  echo "Scrub the hit before committing/publishing (or, if it is a deliberate false positive," >&2
  echo "narrow the pattern). Set GBENCH_LEAK_PATTERNS to also scan the private codename denylist." >&2
  exit 1
fi

# File CONTENT is not the only leak vector: commit AUTHOR/COMMITTER metadata across all history
# (which `git grep` cannot see) can carry a corporate email or name. This can only be removed by a
# history rewrite, so surface it here as a hard blocker before a repo goes public.
meta=$(git log --all --format='%ae%n%ce%n%an%n%cn' 2>/dev/null | sort -u | grep -iE "$pattern" || true)
if [ -n "$meta" ]; then
  echo "❌ leak-gate: internal-identifier pattern in git commit author/committer METADATA:" >&2
  echo "$meta" >&2
  echo "" >&2
  echo "A file scrub cannot remove this - it requires a history rewrite (fresh/orphan history or" >&2
  echo "git filter-repo --mailmap) before publishing." >&2
  exit 1
fi

echo "✅ leak-gate: no internal-identifier / secret patterns in tracked files or commit metadata."
