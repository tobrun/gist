# MCP

## Browser Tools

- Download Chrome extension from https://browsertools.agentdesk.ai/
- Install the Browser‑Tools MCP server into Claude Code:

```
claude mcp add browser-tools -s user \
  -- npx -y @agentdeskai/browser-tools-mcp@1.2.1
```
- Run Server
```
npx -y @agentdeskai/browser-tools-server@1.2.1
```
- Open Chrome DevTools and look for the BrowserTools panel there to start capturing logs and browser context.
- You should see an indication that devtools is debugging your session

## JIRA

Replace inline variables

```json
"mcpServers": {
    "mcp-atlassian": {
      "command": "uvx",
      "args": [
        "mcp-atlassian",
        "--jira-url=https://{{DOMAIN}}.atlassian.net/",
        "--jira-username={{EMAIL}}",
        "--jira-token={{TOKEN}}"
      ]
    }
}
```

## Circle-CI

Replace inline variables

```json
"mcpServers": {
    "circleci-mcp-server": {
      "command": "npx",
      "args": [
        "-y",
        "@circleci/mcp-server-circleci"
      ],
      "env": {
        "CIRCLECI_TOKEN": "{{TOKEN}}"
      }
    }
}
```