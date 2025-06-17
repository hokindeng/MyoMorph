#!/bin/bash
"""
Upload Large Files to S3: s3://myo-data/MyoMorph
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
🚀 MyoMorph S3 Upload Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    all             Upload all large files
    checkpoints     Upload model checkpoints
    datasets        Upload datasets
    deps            Upload dependencies
    models          Upload trained models
    results         Upload generated results
    
Options:
    --dry-run       Show what would be uploaded without uploading
    --verbose       Show detailed output
    --profile PROF  Use specific AWS profile (default: default)
    --help          Show this help message

Examples:
    $0 all                          # Upload everything
    $0 checkpoints --dry-run        # Preview checkpoint uploads
    $0 datasets --verbose           # Upload datasets with details
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

# Function to upload directory with sync
upload_directory() {
    local local_dir="$1"
    local s3_path="$2"
    local description="$3"
    
    if [ ! -d "$local_dir" ]; then
        print_warning "Directory not found: $local_dir"
        return 0
    fi
    
    print_status "Uploading $description..."
    print_status "  Local: $local_dir"
    print_status "  S3: $s3_path"
    
    local aws_cmd="aws s3 sync \"$local_dir\" \"$s3_path\" --profile $AWS_PROFILE"
    
    if [ "$DRY_RUN" = true ]; then
        aws_cmd="$aws_cmd --dryrun"
        print_warning "DRY RUN MODE - No files will be uploaded"
    fi
    
    if [ "$VERBOSE" = true ]; then
        aws_cmd="$aws_cmd --progress"
    fi
    
    # Add common sync options
    aws_cmd="$aws_cmd --delete --exclude '.git/*' --exclude '__pycache__/*'"
    
    if eval $aws_cmd; then
        print_success "✅ $description uploaded successfully"
    else
        print_error "❌ Failed to upload $description"
        return 1
    fi
}

# Function to upload large files
upload_file() {
    local local_file="$1"
    local s3_path="$2"
    local description="$3"
    
    if [ ! -f "$local_file" ]; then
        print_warning "File not found: $local_file"
        return 0
    fi
    
    print_status "Uploading $description..."
    print_status "  Local: $local_file"
    print_status "  S3: $s3_path"
    
    local aws_cmd="aws s3 cp \"$local_file\" \"$s3_path\" --profile $AWS_PROFILE"
    
    if [ "$DRY_RUN" = true ]; then
        aws_cmd="$aws_cmd --dryrun"
        print_warning "DRY RUN MODE - No files will be uploaded"
    fi
    
    if eval $aws_cmd; then
        print_success "✅ $description uploaded successfully"
    else
        print_error "❌ Failed to upload $description"
        return 1
    fi
}

# Upload checkpoints
upload_checkpoints() {
    print_status "🔄 Uploading Model Checkpoints..."
    
    # Upload all checkpoint directories
    upload_directory "./checkpoints" "$S3_BUCKET/checkpoints" "Model Checkpoints"
    
    # Upload any standalone model files
    for ext in ckpt pth pt bin safetensors h5; do
        find . -name "*.$ext" -not -path "./.git/*" -not -path "./myo_*/*" | while read file; do
            s3_path="$S3_BUCKET/models/$(basename "$file")"
            upload_file "$file" "$s3_path" "Model file: $(basename "$file")"
        done
    done
}

# Upload datasets
upload_datasets() {
    print_status "📊 Uploading Datasets..."
    
    upload_directory "./datasets" "$S3_BUCKET/datasets" "All Datasets"
    
    # Upload any zip/tar files that might be datasets
    for ext in zip tar tar.gz tar.xz; do
        find . -name "*.$ext" -not -path "./.git/*" -not -path "./myo_*/*" | while read file; do
            s3_path="$S3_BUCKET/datasets/$(basename "$file")"
            upload_file "$file" "$s3_path" "Dataset archive: $(basename "$file")"
        done
    done
}

# Upload dependencies
upload_deps() {
    print_status "📦 Uploading Dependencies..."
    
    upload_directory "./deps" "$S3_BUCKET/deps" "Dependencies and Pre-trained Models"
}

# Upload results
upload_results() {
    print_status "📈 Uploading Results..."
    
    upload_directory "./results" "$S3_BUCKET/results" "Generated Results"
    
    # Upload video files
    for ext in mp4 avi gif; do
        find . -name "*.$ext" -not -path "./.git/*" -not -path "./myo_*/*" | while read file; do
            s3_path="$S3_BUCKET/results/videos/$(basename "$file")"
            upload_file "$file" "$s3_path" "Video: $(basename "$file")"
        done
    done
    
    # Upload numpy arrays
    find . -name "*.npy" -not -path "./.git/*" -not -path "./myo_*/*" | while read file; do
        s3_path="$S3_BUCKET/results/motions/$(basename "$file")"
        upload_file "$file" "$s3_path" "Motion data: $(basename "$file")"
    done
}

# Upload all categories
upload_all() {
    print_status "🚀 Starting complete upload to S3..."
    print_status "Bucket: $S3_BUCKET"
    echo
    
    upload_checkpoints
    echo
    upload_datasets  
    echo
    upload_deps
    echo
    upload_results
    echo
    
    print_success "🎉 Complete upload finished!"
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
            all|checkpoints|datasets|deps|models|results)
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
        COMMAND="all"
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
            upload_all
            ;;
        checkpoints)
            upload_checkpoints
            ;;
        datasets)
            upload_datasets
            ;;
        deps)
            upload_deps
            ;;
        models)
            upload_checkpoints  # Same as checkpoints
            ;;
        results)
            upload_results
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