#!/bin/bash
# Budget alert setup - warns when 250 USD of free-trial credit is consumed

set -e

PROJECT_ID=$(gcloud config get-value project)

# Find the billing account tied to the current project
BILLING_ACCOUNT_ID=$(
  gcloud beta billing projects describe "$PROJECT_ID" \
         --format='value(billingAccountName)' | awk -F/ '{print $2}'
)

if [[ -n "$BILLING_ACCOUNT_ID" ]]; then
  gcloud billing budgets create \
    --billing-account="$BILLING_ACCOUNT_ID" \
    --display-name="trial-credit-guard" \
    --budget-amount=250USD \
    --threshold-rule="percent=1" || true
  echo "💰  Budget alert created on account $BILLING_ACCOUNT_ID"
else
  echo "⚠️  Could not detect a billing account – budget alert skipped."
fi