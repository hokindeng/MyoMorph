#!/bin/bash
"""
Download Large Files from S3: s3://myo-data/MyoMorph
Handles checkpoints, datasets, models, and dependencies
"""

set -e  # Exit on any error

# Configuration
S3_BUCKET="s3://myo-data/MyoMorph"
AWS_PROFILE="${AWS_PROFILE:-default}"
DRY_RUN=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
📥 MyoMorph S3 Download Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    all             Download all large files
    checkpoints     Download model checkpoints
    datasets        Download datasets
    deps            Download dependencies
    models          Download trained models
    results         Download generated results
    list            List available files in S3
    
Options:
    --dry-run       Show what would be downloaded without downloading
    --verbose       Show detailed output
    --profile PROF  Use specific AWS profile (default: default)
    --help          Show this help message

Examples:
    $0 all                          # Download everything
    $0 checkpoints --dry-run        # Preview checkpoint downloads
    $0 datasets --verbose           # Download datasets with details
    $0 list                         # List all files in S3
    $0 --profile myprofile all      # Use specific AWS profile

S3 Bucket: $S3_BUCKET
EOF
}

# Function to check if AWS CLI is configured
check_aws_config() {
    if ! command -v aws >/dev/null 2>&1; then
        print_error "AWS CLI is not installed. Install with: pip install awscli"
        exit 1
    fi
    
    if ! aws sts get-caller-identity --profile $AWS_PROFILE >/dev/null 2>&1; then
        print_error "AWS credentials not configured for profile: $AWS_PROFILE"
        print_status "Configure with: aws configure --profile $AWS_PROFILE"
        exit 1
    fi
    
    print_success "AWS CLI configured for profile: $AWS_PROFILE"
}

# Function to download directory with sync
download_directory() {
    local s3_path="$1"
    local local_dir="$2"
    local description="$3"
    
    print_status "Downloading $description..."
    print_status "  S3: $s3_path"
    print_status "  Local: $local_dir"
    
    # Create local directory if it doesn't exist
    mkdir -p "$local_dir"
    
    local aws_cmd="aws s3 sync \"$s3_path\" \"$local_dir\" --profile $AWS_PROFILE"
    
    if [ "$DRY_RUN" = true ]; then
        aws_cmd="$aws_cmd --dryrun"
        print_warning "DRY RUN MODE - No files will be downloaded"
    fi
    
    if [ "$VERBOSE" = true ]; then
        aws_cmd="$aws_cmd --progress"
    fi
    
    # Add common sync options
    aws_cmd="$aws_cmd --exclude '.git/*' --exclude '__pycache__/*'"
    
    if eval $aws_cmd; then
        print_success "✅ $description downloaded successfully"
    else
        print_error "❌ Failed to download $description"
        return 1
    fi
}

# Function to download single file
download_file() {
    local s3_path="$1"
    local local_file="$2"
    local description="$3"
    
    print_status "Downloading $description..."
    print_status "  S3: $s3_path"
    print_status "  Local: $local_file"
    
    # Create local directory if it doesn't exist
    mkdir -p "$(dirname "$local_file")"
    
    local aws_cmd="aws s3 cp \"$s3_path\" \"$local_file\" --profile $AWS_PROFILE"
    
    if [ "$DRY_RUN" = true ]; then
        aws_cmd="$aws_cmd --dryrun"
        print_warning "DRY RUN MODE - No files will be downloaded"
    fi
    
    if eval $aws_cmd; then
        print_success "✅ $description downloaded successfully"
    else
        print_error "❌ Failed to download $description"
        return 1
    fi
}

# Download checkpoints
download_checkpoints() {
    print_status "🔄 Downloading Model Checkpoints..."
    
    download_directory "$S3_BUCKET/checkpoints" "./checkpoints" "Model Checkpoints"
    download_directory "$S3_BUCKET/models" "./models" "Additional Model Files"
}

# Download datasets
download_datasets() {
    print_status "📊 Downloading Datasets..."
    
    download_directory "$S3_BUCKET/datasets" "./datasets" "All Datasets"
}

# Download dependencies
download_deps() {
    print_status "📦 Downloading Dependencies..."
    
    download_directory "$S3_BUCKET/deps" "./deps" "Dependencies and Pre-trained Models"
}

# Download results
download_results() {
    print_status "📈 Downloading Results..."
    
    download_directory "$S3_BUCKET/results" "./results" "Generated Results"
}

# List S3 contents
list_s3_contents() {
    print_status "📋 Listing S3 Contents..."
    
    if ! aws s3 ls "$S3_BUCKET" --recursive --profile $AWS_PROFILE --human-readable --summarize; then
        print_error "Failed to list S3 contents"
        return 1
    fi
}

# Download all categories
download_all() {
    print_status "📥 Starting complete download from S3..."
    print_status "Bucket: $S3_BUCKET"
    echo
    
    download_checkpoints
    echo
    download_datasets  
    echo
    download_deps
    echo
    download_results
    echo
    
    print_success "🎉 Complete download finished!"
}

# Main script logic
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --profile)
                AWS_PROFILE="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            all|checkpoints|datasets|deps|models|results|list)
                COMMAND="$1"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Default command if none specified
    if [ -z "$COMMAND" ]; then
        print_warning "No command specified. Use --help for usage information."
        show_usage
        exit 1
    fi
    
    # Print configuration
    echo "🔧 Configuration:"
    echo "   S3 Bucket: $S3_BUCKET"
    echo "   AWS Profile: $AWS_PROFILE"
    echo "   Command: $COMMAND"
    echo "   Dry Run: $DRY_RUN"
    echo "   Verbose: $VERBOSE"
    echo
    
    # Check AWS configuration
    check_aws_config
    echo
    
    # Execute command
    case $COMMAND in
        all)
            download_all
            ;;
        checkpoints)
            download_checkpoints
            ;;
        datasets)
            download_datasets
            ;;
        deps)
            download_deps
            ;;
        models)
            download_checkpoints  # Same as checkpoints
            ;;
        results)
            download_results
            ;;
        list)
            list_s3_contents
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 