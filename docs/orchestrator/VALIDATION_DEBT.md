# Validation Debt Ledger

## Purpose

Record known validation failures outside the active task.

## Rules

Validation debt never excuses a current-task failure. Each entry names owner, scope, unblock condition, and sanitized evidence. Do not include secrets, tokens, raw production data, or private evidence.

## Entries

No validation debt is recorded for the completed canonical adoption.

## Update format

When debt exists, add ID, date, command, sanitized failure, scope, owner, current-task impact, unblock condition, and evidence path.
