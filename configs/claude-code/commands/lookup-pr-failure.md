Diagnose all CI failures (GitHub Actions & CircleCI platform) for the current pull request.

Use gh pr view <pr_number> --json statusCheckRollup to find all checks with a conclusion of FAILURE.

For GitHub Actions failures, parse the run_id from the detailsUrl and get the log with gh run view --log.

For CircleCI failures, use your custom circle-ci-mcp tool with the job's detailsUrl to retrieve the failure logs.

Analyze all retrieved logs to pinpoint the root cause for each failure.

Report back with a clean summary listing only the failed job names and their specific error messages.