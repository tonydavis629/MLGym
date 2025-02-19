# Changelog

All notable changes to MLGym will be documented in this file.

## [0.1] - First Release

**Date: 14 February 2025**

### Added

- Added `backend` submodule for clear separation between model providers.
- Support for API based models using LiteLLM.
- Added `ToolsConfig` and `ToolHandler` to manage the functionality for Tools and decouple them from the agent code.

### Changed

- Agent configuration path should be specified using `--agent_config_path` instead of `config_file`.
- Agent configuration files accept a new section `tools`. All the tool related arguments such as `command_files`, `parser` etc. can be specified in the new section.
- Tool location has changed from `MLGYM/configs/commands/` to `MLGYM/tools/`. Please update your agent configuration files.
- Demostrations location has changed from `MLGYM/configs/demonstrations/` to `MLGYM/demonstrations`. Please update your agent configuration files.

### Removed

- Subroutines are now removed following the [SWE-Agent 1.0](https://swe-agent.com/latest/installation/migration/) update.
