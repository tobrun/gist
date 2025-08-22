# MCP

## Browser Tools

- Download Chrome extension from https://browsertools.agentdesk.ai/
- Install the Browser‑Tools MCP server into Claude Code:

```
claude mcp add browser-tools -s user \
  -- npx -y @agentdeskai/browser-tools-mcp@1.2.1
```
- npx -y @agentdeskai/browser-tools-server@1.2.1
- Open Chrome DevTools and look for the BrowserTools panel there to start capturing logs and browser context.
