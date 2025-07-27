#!/bin/bash

# Log Level Control Script for Playlista
# Usage: ./log_level.sh [debug|info|warning|help]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

# Function to change log level
change_log_level() {
    local level=$1
    local signal=""
    
    case $level in
        "debug"|"DEBUG")
            signal="SIGUSR1"
            ;;
        "info"|"INFO")
            signal="SIGUSR2"
            ;;
        "warning"|"WARNING")
            signal="SIGTERM"
            ;;
        *)
            print_error "Invalid log level: $level"
            echo "Valid levels: debug, info, warning"
            exit 1
            ;;
    esac
    
    print_info "Changing log level to: ${level^^}"
    
    # Try to send signal to the playlista process
    if docker compose exec playlista bash -c "kill -$signal 1" 2>/dev/null; then
        print_status "Log level changed to ${level^^}"
    else
        print_error "Failed to change log level. Is playlista running?"
        exit 1
    fi
}

# Function to show current log level
show_current_level() {
    print_info "Current log level: $(docker compose exec playlista bash -c 'echo $LOG_LEVEL' 2>/dev/null || echo 'INFO')"
}

# Function to show help
show_help() {
    echo "🎵 Playlista Log Level Control"
    echo ""
    echo "Usage:"
    echo "  ./log_level.sh debug    - Set log level to DEBUG (verbose)"
    echo "  ./log_level.sh info     - Set log level to INFO (normal)"
    echo "  ./log_level.sh warning  - Set log level to WARNING (quiet)"
    echo "  ./log_level.sh current  - Show current log level"
    echo "  ./log_level.sh help     - Show this help"
    echo ""
    echo "Examples:"
    echo "  ./log_level.sh debug    # Make logs more verbose"
    echo "  ./log_level.sh warning  # Make logs less verbose"
    echo ""
    echo "Note: This script sends signals to the running playlista process."
    echo "      Make sure playlista is running before using this script."
}

# Main script logic
case "${1:-help}" in
    "debug"|"DEBUG")
        change_log_level "debug"
        ;;
    "info"|"INFO")
        change_log_level "info"
        ;;
    "warning"|"WARNING")
        change_log_level "warning"
        ;;
    "current"|"status")
        show_current_level
        ;;
    "help"|"-h"|"--help"|"")
        show_help
        ;;
    *)
        print_error "Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 