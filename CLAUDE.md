# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Local Development

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the application:**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

## Architecture

This project is a FastAPI application that acts as a proxy and load balancer for the Google Gemini API.

- **`app/main.py`**: The main entry point of the application.
- **`app/config/`**: Contains application configuration.
- **`app/router/`**: Defines the API routes for Gemini, OpenAI, and other functionalities. The main routes are in `app/router/routes.py`.
- **`app/service/`**: Holds the core business logic.
  - `app/service/chat/`: Handles chat-related services.
  - `app/service/key/`: Manages API key rotation and validation.
- **`app/database/`**: Manages database connections and models.
- **`app/middleware/`**: Contains FastAPI middleware for request handling.
- **`app/templates/`**: HTML templates for the web interface.
- **`app/static/`**: Static assets like CSS and JavaScript.

## Dependencies

- **`fastapi`**: The web framework used for building the API.
- **`uvicorn`**: The ASGI server for running the FastAPI application.
- **`google-genai`**: The official Python SDK for the Gemini API.
- **`openai`**: The official Python library for the OpenAI API.
- **`sqlalchemy`**: The SQL toolkit and Object-Relational Mapper (ORM).
- **`httpx`**: An HTTP client for making requests to external APIs.
- **`apscheduler`**: For running scheduled tasks, like checking key status.
