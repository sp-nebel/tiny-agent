# tiny-agent

A minimal local coding agent for CPU-only machines, built on Ollama. Before changing code, read the project skills in `.claude/skills/`: `working-on-tiny-agent` (map and invariants), `cache-discipline`, `model-facing-text` and `verifying-tiny-agent`.

## Keep the changelog current on every commit

`README.md` ends with a `## Changelog` section. Every commit updates it **in the same commit**:

- Add an entry at the top, under a `### YYYY-MM-DD` heading for the commit date. Create the heading if it's missing.
- Format: `- **Short name** (\`hash\`): what changed, from the user's point of view, in one to three sentences.`
- A commit can't contain its own hash, so leave the `(\`hash\`)` part off a new entry. When you next edit the changelog, fill in the hashes of any entries that are missing them (`git log --oneline`).
- Small related commits (docs tweaks, follow-up fixes) can share one entry or be added to an existing entry's hash list.
- Merge commits get no entry of their own. Name the PR in the entry for the change they merged.
- If an entry becomes wrong (the feature was reverted or changed), fix it rather than leaving it stale.
