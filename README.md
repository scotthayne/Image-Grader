# Streamlit Dashboard Modal Deployment

A template for deploying Streamlit applications to Modal cloud platform. This project provides a minimal setup for taking any Streamlit dashboard and deploying it to Modal's serverless infrastructure.

## Project Structure

```
.
├── README.md
├── instructions.md
├── app.py              # Your Streamlit application
└── serve_streamlit.py  # Modal deployment configuration
```

## Quick Start

1. Place your Streamlit app in `app.py`
2. Follow `instructions.md` for deployment steps
3. Access your app at `https://[username]--[app-name]-run.modal.run`

## Requirements

- Python 3.11+
- uv (Python package manager)
- Modal account
