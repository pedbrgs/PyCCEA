# Contributing Guide

We welcome contributions to this project! Whether you're fixing a bug, improving documentation, or introducing a new feature, your help is appreciated. Please take a moment to review the guidelines below before submitting a pull request (PR).

## :rocket: How to contribute via pull requests

To contribute effectively and help maintain code quality, follow these steps:

### :fork_and_knife: 1. Fork the repository
Start by forking the repository to your GitHub account (e.g., `pyccea/pyccea`).

### :seedling: 2. Create a branch
Create a new branch in your forked repository for your changes. Use a descriptive and concise name:
- `fix-issue-123`
- `add-feature-xyz`
- `improve-docs`

### :hammer_and_wrench: 3. Make your changes
Implement your improvements or bug fixes. Ensure your code:
- Follows the project's style and architecture.
- Is well-organized and readable.
- Avoids introducing warnings or breaking existing functionality.

### :white_check_mark: 4. Test locally
Before submitting a PR:
- Run existing tests and verify they pass.
- Add new tests for new functionality where appropriate.
- Ensure your code works in the expected environments.

### :memo: 5. Commit your changes
Write clear and descriptive commit messages using the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This helps maintain a consistent history and supports semantic versioning. Use types like feat, fix, chore, docs, or test to describe the nature of your changes.

Examples:

- feat: add new evaluation metric for classification tasks
- fix: resolve crash when loading empty dataset


### :mailbox_with_mail: 6. Open a pull request
Submit your pull request to the main branch of the original repository. Your PR should include:
- A clear and descriptive title.
- A summary of the changes and their purpose.
- Any relevant context or issue references (e.g., `Closes #123`).

---

## :test_tube: Code style

We follow these general practices:

- **PEP8 compliance**: Use consistent indentation (4 spaces), limit lines to 79 characters where possible, and follow naming conventions.
- **Type annotations**: Use type hints to improve code clarity and support static analysis.
- **Docstrings**: Add docstrings to public modules, classes, and functions using the [Numpy Python Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html).
- **Logging**: Prefer `logging` over `print` for production/debug output.
- **Avoid large commits**: Break changes into logical units; it helps reviewers and future maintainers.
- **Stay dry**: Avoid duplicating logic; reuse functions or create helpers where needed.

---

## :handshake: Acknowledgments

Thank you for taking the time to contribute to this project! We truly appreciate your effort and interest in helping improve the PyCCEA package.

For questions or help, feel free to open a discussion or reach out in the relevant issue thread.
