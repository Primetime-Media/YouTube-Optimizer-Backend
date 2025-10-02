#!/bin/bash

# ============================================================================
# YouTube Optimizer - Quick Setup Script
# Version: 2.0
# ============================================================================
#
# This script automates the complete database setup process.
# Run with: bash quick_setup.sh
#
# What it does:
# 1. Creates the database
# 2. Applies the complete schema
# 3. Creates indexes
# 4. Sets up triggers
# 5. Verifies installation
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DB_NAME="youtube_optimizer"
DB_USER="postgres"
DB_PASSWORD=""  # Set this or use .pgpass
BACKUP_DIR="backups/$(date +%Y%m%d)"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

check_postgres() {
    print_header "Checking PostgreSQL Installation"
    
    if ! command -v psql &> /dev/null; then
        print_error "PostgreSQL not found. Please install PostgreSQL 15+."
        exit 1
    fi
    
    print_success "PostgreSQL found: $(psql --version)"
    
    # Check if PostgreSQL is running
    if ! pg_isready -q; then
        print_error "PostgreSQL is not running. Please start it first."
        print_warning "Try: sudo systemctl start postgresql"
        exit 1
    fi
    
    print_success "PostgreSQL is running"
}

backup_existing_db() {
    print_header "Checking for Existing Database"
    
    if psql -U "$DB_USER" -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        print_warning "Database $DB_NAME already exists!"
        read -p "Do you want to backup and recreate it? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            mkdir -p "$BACKUP_DIR"
            
            BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_$(date +%H%M%S).sql"
            print_warning "Backing up to: $BACKUP_FILE"
            
            pg_dump -U "$DB_USER" "$DB_NAME" > "$BACKUP_FILE"
            
            if [ $? -eq 0 ]; then
                print_success "Backup completed"
                
                # Drop database
                psql -U "$DB_USER" -c "DROP DATABASE $DB_NAME;"
                print_success "Old database dropped"
            else
                print_error "Backup failed. Aborting."
                exit 1
            fi
        else
            print_error "Setup cancelled. Existing database preserved."
            exit 0
        fi
    else
        print_success "No existing database found"
    fi
}

create_database() {
    print_header "Creating Database"
    
    psql -U "$DB_USER" <<EOF
CREATE DATABASE $DB_NAME
  WITH 
  ENCODING = 'UTF8'
  LC_COLLATE = 'en_US.UTF-8'
  LC_CTYPE = 'en_US.UTF-8'
  TEMPLATE = template0;
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Database created successfully"
    else
        print_error "Failed to create database"
        exit 1
    fi
}

apply_schema() {
    print_header "Applying Database Schema"
    
    if [ ! -f "database/schema.sql" ]; then
        print_error "Schema file not found at database/schema.sql"
        print_warning "Please ensure you're running this from the project root directory"
        exit 1
    fi
    
    psql -U "$DB_USER" -d "$DB_NAME" -f database/schema.sql
    
    if [ $? -eq 0 ]; then
        print_success "Schema applied successfully"
    else
        print_error "Failed to apply schema"
        exit 1
    fi
}

verify_installation() {
    print_header "Verifying Installation"
    
    # Check table count
    TABLE_COUNT=$(psql -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    " | xargs)
    
    print_success "Tables created: $TABLE_COUNT"
    
    if [ "$TABLE_COUNT" -lt 5 ]; then
        print_error "Expected at least 5 tables, found $TABLE_COUNT"
        exit 1
    fi
    
    # List tables
    echo ""
    echo "Tables in database:"
    psql -U "$DB_USER" -d "$DB_NAME" -c "\dt" | grep public
    
    # Check indexes
    INDEX_COUNT=$(psql -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT COUNT(*) 
        FROM pg_indexes 
        WHERE schemaname = 'public';
    " | xargs)
    
    print_success "Indexes created: $INDEX_COUNT"
    
    # Check views
    VIEW_COUNT=$(psql -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT COUNT(*) 
        FROM information_schema.views 
        WHERE table_schema = 'public';
    " | xargs)
    
    print_success "Views created: $VIEW_COUNT"
}

test_connection() {
    print_header "Testing Database Connection"
    
    # Test connection string from .env if it exists
    if [ -f ".env" ]; then
        DB_URL=$(grep DATABASE_URL .env | cut -d '=' -f2)
        if [ ! -z "$DB_URL" ]; then
            print_success "Found DATABASE_URL in .env"
            
            # Extract connection details and test
            python3 -c "
import os
from sqlalchemy import create_engine

try:
    engine = create_engine('$DB_URL')
    conn = engine.connect()
    conn.close()
    print('✓ Connection successful using DATABASE_URL')
    exit(0)
except Exception as e:
    print(f'✗ Connection failed: {e}')
    exit(1)
            " && print_success "Connection test passed" || print_error "Connection test failed"
        fi
    else
        print_warning ".env file not found. Create it before running the application."
    fi
}

create_env_example() {
    print_header "Creating .env.example"
    
    if [ ! -f ".env.example" ]; then
        cat > .env.example <<EOF
# Database Configuration
DATABASE_URL=postgresql://${DB_USER}:password@localhost:5432/${DB_NAME}
DB_ENCRYPTION_KEY=your-32-byte-fernet-key-here

# API Keys
ANTHROPIC_API_KEY=your-anthropic-api-key
SERPAPI_API_KEY=your-serpapi-api-key

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/callback

# Application Settings
SECRET_KEY=your-secret-key-min-32-characters-long
ENVIRONMENT=development
LOG_LEVEL=INFO
SCHEDULER_SECRET_KEY=your-scheduler-secret-key

# Optional: External Services
REDIS_URL=redis://localhost:6379/0
EOF
        print_success ".env.example created"
        print_warning "Copy it to .env and fill in your actual credentials"
    else
        print_success ".env.example already exists"
    fi
}

print_next_steps() {
    print_header "Setup Complete!"
    
    echo ""
    echo -e "${GREEN}✓ Database created: $DB_NAME${NC}"
    echo -e "${GREEN}✓ Schema applied successfully${NC}"
    echo -e "${GREEN}✓ All tables and indexes created${NC}"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo ""
    echo "1. Copy .env.example to .env:"
    echo -e "   ${GREEN}cp .env.example .env${NC}"
    echo ""
    echo "2. Edit .env and add your API keys and credentials"
    echo ""
    echo "3. Generate a Fernet encryption key:"
    echo -e "   ${GREEN}python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'${NC}"
    echo ""
    echo "4. Install Python dependencies:"
    echo -e "   ${GREEN}pip install -r requirements.txt${NC}"
    echo ""
    echo "5. Run the application:"
    echo -e "   ${GREEN}python main.py${NC}"
    echo ""
    echo "6. Test the health endpoint:"
    echo -e "   ${GREEN}curl http://localhost:8000/health${NC}"
    echo ""
    echo -e "${YELLOW}Database Connection String:${NC}"
    echo -e "${GREEN}postgresql://${DB_USER}@localhost:5432/${DB_NAME}${NC}"
    echo ""
    echo -e "${YELLOW}Documentation:${NC}"
    echo "- Full setup guide: README.md"
    echo "- Migration guide: DATABASE_MIGRATION.md"
    echo "- API docs (when running): http://localhost:8000/docs"
    echo ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    clear
    
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  YouTube Optimizer - Quick Setup v2.0     ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Run setup steps
    check_postgres
    backup_existing_db
    create_database
    apply_schema
    verify_installation
    test_connection
    create_env_example
    print_next_steps
    
    exit 0
}

# Run main function
main "$@"
