# Security Policy

## Supported Scope

Security reports are welcome for all active code in this repository.

## Reporting a Vulnerability

Please do not open public issues for vulnerabilities.

Report privately to the maintainers with:

- Vulnerability summary
- Impact
- Reproduction steps
- Suggested fix if available

## Secret Management Requirements

- Never hardcode API keys or tokens in source files.
- Use environment variables for provider credentials.
- Rotate credentials immediately if exposure is suspected.

Recommended environment variables:

- `OPENAI_API_KEY`
- `HUGGINGFACE_TOKEN`

## Data Handling

- Do not commit real patient-identifying or regulated medical data.
- Use synthetic or anonymized test datasets.
- Remove sensitive outputs before public release.

## Disclosure Process

1. Maintainers acknowledge report.
2. Issue is triaged and validated.
3. Fix is prepared and tested.
4. Public disclosure occurs after remediation is available.

