# Example Plugin

This is an example plugin that demonstrates the AI Learning Accelerator plugin architecture.

## Features

- **API Endpoints**: Provides `/example/hello` and `/example/status` endpoints
- **Webhook Handlers**: Handles `user.created` and `learning.progress` webhooks
- **Event Handlers**: Responds to system startup and shutdown events
- **Configuration**: Supports configurable greeting message

## Configuration

```json
{
  "greeting": "Hello",
  "enabled": true
}
```

## API Endpoints

### GET /example/hello

Returns a greeting message.

**Response:**
```json
{
  "message": "Hello from Example Plugin!",
  "call_count": 1,
  "status": "active"
}
```

### POST /example/hello

Returns a personalized greeting message.

**Request:**
```json
{
  "name": "John"
}
```

**Response:**
```json
{
  "message": "Hello John from Example Plugin!",
  "call_count": 2,
  "status": "active"
}
```

### GET /example/status

Returns plugin health status.

**Response:**
```json
{
  "status": "active",
  "healthy": true,
  "last_error": null,
  "uptime": "2024-01-01T12:00:00",
  "call_count": 3,
  "config": {
    "greeting": "Hello",
    "enabled": true
  }
}
```

## Webhook Handlers

### user.created

Handles new user creation events.

### learning.progress

Handles learning progress updates.

## Installation

1. Place this plugin directory in the `plugins/` folder
2. Use the Plugin Management API to load and start the plugin
3. The plugin will be automatically discovered and available for loading