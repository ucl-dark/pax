---
name: "\U0001F41B Bug Report"
about: Submit a bug report to help us improve Pax. If this doesn’t look right,
  choose a different type.
title: "[BUG]"
labels: 'bug'
---

body:
- type: markdown
  attributes:
    value: >
      ## Thank you for helping us improve PAX!

      * Please first verify that your issue is not already reported using the
      [Issue search][issue search].

      [issue search]: https://github.com/akbir/pax/search?q=is%3Aissue&type=issues
- type: textarea
  attributes:
    label: Description
    description: >-
      A concise description of the bug, preferably including self-contained
      code to reproduce the issue.
    placeholder: |
      Text may use markdown formatting.
      ```python
      # for codeblocks, use triple backticks
      ```
  validations:
    required: true
- type: input
  attributes:
    label: What pax version are you using?
    placeholder: For example jax v0.1.0b+3c35a4e
- type: input
  attributes:
    label: What jax/jaxlib version are you using?
    placeholder: For example jax v0.3.0, jaxlib v0.3.0
- type: input
  attributes:
    label: Which accelerator(s) are you using?
    placeholder: CPU/GPU/TPU
- type: input
  attributes:
    label: Additional system info
    placeholder: Python version, OS (Linux/Mac/Windows/WSL), etc.
- type: textarea
  attributes:
    label: NVIDIA GPU info
    description: >-
      If you are using an NVIDIA GPU, what is the output of the `nvidia-smi` command?
    placeholder: |
      ```
      ...
      ```
