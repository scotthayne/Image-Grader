# Streamlit Dashboard + Modal Deployment

This templates develops, spins up, and deploys a Streamlit applications to Modal cloud platform. This project provides a minimal setup for taking any Streamlit dashboard and deploying it to Modal's serverless infrastructure.

## Overview

This template will create a cool Streamlit dashboard before deploying it to Modal, allowing you to see how Memex:
- Codes an app
- Starts a Streamlit server
- Opens and iterates on a Streamlit app
- Deploys the app to Modal (including giving you instructions of how to set up Modal if they haven't already)

## Potential app functionality expansions to explore

Here are some ideas of how to expand this template after you get it up and running:
- Generate a new Streamlit app based on your needs
- Iterate on your app to add or remove features
- Deploy it on your Modal instance to make it accessible to others

## Requirements

- Python 3.11+
- uv (Python package manager)
- Modal account (Memex can provide instructions of how to set this up)

## Quick Start

Just ask Memex to run this app locally and it will take care of the rest! If you run into any errors, just point Memex to fix them.

If you’d like to set up the environment and dependencies manually, follow these steps:

1. Place your Streamlit app in `app.py`
2. Follow `instructions.md` for deployment steps
3. Access your app at `https://[username]--[app-name]-run.modal.run`

## Development

See Rules for AI (rendered from .memex/rules.md) for detailed development guidelines Memex will follow, including:
- Complete setup instructions
- Model-specific parameters
- Error handling
- Potential improvements
- Development workflow

You can ask Memex to update rules.md to reflect your project needs as you expand it, or set it as part of your Custom Instructions so that it does it automatically after important steps.

## License

[Streamlit](https://github.com/streamlit/streamlit) is Open Source, and [Modal](https://modal.com/) has a free trial available.
