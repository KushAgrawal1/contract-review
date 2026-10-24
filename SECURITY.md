# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's
"Report a vulnerability" option on the Security tab of this repository.

## Secrets

API keys and database credentials are read from environment variables
(see `.env.example`). Never commit a real `.env` file.

## Contract data

Do not upload confidential or personal contracts to a public deployment.
The demo is intended for public CUAD contracts only.
