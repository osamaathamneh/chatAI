# LLMonDBPGAI Frontend

Frontend application for Natural Language to SQL interface.

## Overview

This is a static frontend application that communicates with the LLMonDBPGAI Backend API. It provides a user-friendly interface for:
- Semantic database search
- Natural language to SQL generation
- SQL query execution
- Query history management
- Model evaluation

## Project Structure

```
LLMonDBPGAI_Frontend/
├── src/
│   ├── index.html          # Main search interface
│   ├── fast_chat.html      # Fast chat with template matching
│   ├── evaluation.html     # Model evaluation interface
│   ├── verify_setup.html  # Setup verification tool
│   ├── css/
│   │   └── styles.css      # Application styles
│   ├── js/
│   │   ├── config.js      # Centralized API configuration
│   │   ├── script.js       # Main application logic
│   │   ├── fast_chat.js    # Fast chat functionality
│   │   └── evaluation.js   # Evaluation functionality
│   └── assets/             # Images, icons, etc.
└── README.md
```

## Setup

### Prerequisites

- A web server to serve static files (or use a simple HTTP server)
- Backend API running at `http://localhost:8000`

### Running the Frontend

#### Option 1: Python HTTP Server

```bash
# Navigate to src directory
cd src

# Python 3
python -m http.server 8080

# Or Python 2
python -m SimpleHTTPServer 8080
```

Then open: http://localhost:8080

#### Option 2: Node.js HTTP Server

```bash
# Install http-server globally (if not installed)
npm install -g http-server

# Navigate to src directory
cd src

# Start server
http-server -p 8080
```

#### Option 3: VS Code Live Server

1. Install "Live Server" extension in VS Code
2. Right-click on `index.html`
3. Select "Open with Live Server"

### Configuration

The frontend uses centralized configuration via `src/js/config.js`. The backend API URL is set in one place:

```javascript
window.API_CONFIG = {
    BASE_URL: 'http://localhost:8000'
};
```

If your backend runs on a different port or host, update `BASE_URL` in `src/js/config.js` only. All pages will automatically use the new URL.

**Available Providers:**
- **OpenAI** - Direct OpenAI API access
- **OpenRouter** - Multi-model provider (includes GPT OSS 20B/120B, GPT-4o, Claude models)
- **Ollama** - Local/offline models

## API Endpoints Used

The frontend calls the following backend endpoints:

- `GET /api/health` - Health check
- `GET /api/databases` - List databases
- `GET /api/catalogs` - List semantic catalogs
- `GET /api/models/*` - List models by provider
- `POST /api/search` - Semantic search
- `POST /api/generate-sql` - Generate SQL
- `POST /api/smart-generate-sql` - Smart SQL generation with templates
- `POST /api/execute-sql` - Execute SQL
- `GET /api/history` - Get query history
- `DELETE /api/history` - Clear history
- `GET /api/examples` - Get example queries
- `POST /api/feedback/*` - Feedback endpoints
- `GET /api/templates` - Get query templates
- `POST /api/evaluation/*` - Evaluation endpoints

## Features

### Main Interface (index.html)
- Semantic database search
- SQL generation from natural language
- SQL execution and results display
- Query history
- Example queries
- Navigation to Fast Chat and Evaluation pages

### Fast Chat (fast_chat.html)
- Template-enhanced SQL generation
- Real-time feedback system
- Template matching indicators
- Performance metrics

### Evaluation (evaluation.html)
- Model comparison
- Batch query evaluation
- Accuracy metrics
- Performance charts

### Setup Verification (verify_setup.html)
- Automated connection testing
- Backend health checks
- Configuration validation
- Quick diagnostics

## Development

### File Structure

All HTML files are in the `src/` root directory. CSS and JavaScript are organized in subdirectories for better organization.

### Adding New Features

1. Update the relevant JavaScript file (`script.js`, `fast_chat.js`, or `evaluation.js`)
2. Update HTML if UI changes are needed
3. Update CSS in `css/styles.css` for styling

### CORS Configuration

The backend has CORS enabled for all origins. If you deploy to a different domain, update the CORS configuration in the backend (`src/main.py`).

## Troubleshooting

### API Connection Errors

**Error: Failed to fetch**
- Ensure backend is running at `http://localhost:8000`
- Check browser console for CORS errors
- Verify `API_BASE` constant in JavaScript files

**404 Not Found**
- Verify backend endpoints match frontend API calls
- Check Swagger docs at `http://localhost:8000/docs`

### Static File Loading Issues

**CSS/JS not loading**
- Verify file paths in HTML are correct (relative to HTML file location)
- Check browser console for 404 errors
- Ensure web server is serving files from `src/` directory

## Deployment

For production deployment:

1. Update `API_BASE` to production backend URL
2. Configure CORS in backend to allow production frontend domain
3. Use a production web server (nginx, Apache, etc.)
4. Enable HTTPS for secure API communication

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

Requires modern JavaScript features (async/await, fetch API).

