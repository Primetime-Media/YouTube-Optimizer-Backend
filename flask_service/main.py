"""
Main entry point for the Flask authentication service.
This is the file that Cloud Run will execute.
"""
import os
from app import create_app

app = create_app()

if __name__ == '__main__':
    config = app.config
    port = int(os.environ.get('PORT', config['PORT']))
    app.run(host='0.0.0.0', port=port, debug=config['DEBUG'])