# `CLAUDE.md`

## The Golden Rule

When unsure about implementation details, ALWAYS ask the developer.

## Code Style and Patterns

### Comments

Only include comments to explain hard to understand code, don't add comments to explain your reasoning behind making a change. Comments created should act as inline knowledge that can be easily `grep`ped for.

### Best Practices

- Use best practices of the programming language version used (eg. modern way of formatting strings)
- Produce clean code
- Follow SOLID principles
- Use Design patterns when possible
- Don't duplicate strings
- Apply exisiting paradigms found in the codebase for universal concepts like logging, networking and settings.

## Domain Glossary (Claude, learn these!)

- **Agent**: AI entity with memory, tools, and defined behavior
- **Task**: Workflow definition composed of steps
- **Execution**: Running instance of a task
- **Tool**: Function an agent can call (browser, API, etc.)
- **Session**: Conversation context with memory
- **Entry**: Single interaction within a session

## What AI Must NEVER Do

1. **Never commit secrets** - Use environment variables
2. **Never assume business logic** - Always ask

Remember: We optimize for maintainability over cleverness.
When in doubt, choose the boring solution.