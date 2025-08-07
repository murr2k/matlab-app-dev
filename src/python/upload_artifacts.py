#!/usr/bin/env python3
"""
Upload test artifacts to GitHub Actions or other CI/CD platforms
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

def upload_to_github_actions():
    """Upload artifacts using GitHub Actions environment variables"""
    
    artifacts_dir = Path("test_artifacts")
    if not artifacts_dir.exists():
        print("❌ No test artifacts found")
        return False
    
    # Set GitHub Actions outputs
    if os.getenv('GITHUB_ACTIONS'):
        print("🔧 GitHub Actions environment detected")
        
        # Read pipeline summary
        summary_file = artifacts_dir / "pipeline_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            
            # Set outputs for GitHub Actions
            print(f"::set-output name=success_rate::{summary['test_execution']['success_rate']}")
            print(f"::set-output name=total_tests::{summary['test_execution']['total_test_suites']}")
            print(f"::set-output name=passed_tests::{summary['test_execution']['passed_suites']}")
            print(f"::set-output name=failed_tests::{summary['test_execution']['failed_suites']}")
            print(f"::set-output name=environment::{summary['environment']}")
            
            # Create summary for GitHub Actions
            summary_md = generate_github_summary(summary)
            print(f"::notice::MATLAB Engine Tests completed with {summary['test_execution']['success_rate']:.1f}% success rate")
            
            return True
    
    return False

def generate_github_summary(summary):
    """Generate GitHub Actions job summary"""
    
    success_rate = summary['test_execution']['success_rate']
    status_emoji = "✅" if success_rate == 100 else "⚠️" if success_rate >= 80 else "❌"
    
    md_summary = f"""
# MATLAB Engine API Test Results {status_emoji}

## Summary
- **Environment**: {summary['environment']}
- **Success Rate**: {success_rate:.1f}%
- **Test Suites**: {summary['test_execution']['passed_suites']}/{summary['test_execution']['total_test_suites']} passed

## Artifacts Generated
"""
    
    for artifact in summary.get('artifacts_generated', []):
        md_summary += f"- 📄 `{artifact}`\\n"
    
    if success_rate < 100:
        md_summary += f"""
## ⚠️ Action Required
{summary['test_execution']['failed_suites']} test suite(s) failed. Review the detailed test results.
"""
    
    return md_summary

def create_badge_data():
    """Create badge data for README or status displays"""
    
    artifacts_dir = Path("test_artifacts")
    summary_file = artifacts_dir / "pipeline_summary.json"
    
    if not summary_file.exists():
        return None
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    success_rate = summary['test_execution']['success_rate']
    
    # Determine badge color based on success rate
    if success_rate >= 95:
        color = "brightgreen"
    elif success_rate >= 80:
        color = "yellow"
    else:
        color = "red"
    
    badge_data = {
        "schemaVersion": 1,
        "label": "MATLAB Tests",
        "message": f"{success_rate:.1f}% passing",
        "color": color,
        "cacheSeconds": 3600
    }
    
    # Save badge data
    badge_file = artifacts_dir / "badge.json"
    with open(badge_file, 'w') as f:
        json.dump(badge_data, f, indent=2)
    
    print(f"📊 Badge data generated: {badge_file}")
    return badge_data

if __name__ == "__main__":
    print("🚀 Processing test artifacts...")
    
    # Create badge data
    create_badge_data()
    
    # Upload to CI/CD platform
    if upload_to_github_actions():
        print("✅ Successfully processed artifacts for GitHub Actions")
    else:
        print("ℹ️  No CI/CD platform detected, artifacts saved locally")
    
    # List all generated artifacts
    artifacts_dir = Path("test_artifacts")
    if artifacts_dir.exists():
        print(f"\\n📁 All artifacts in {artifacts_dir}:")
        for artifact in sorted(artifacts_dir.glob("*")):
            size = artifact.stat().st_size
            print(f"  📄 {artifact.name} ({size} bytes)")
    
    print("\\n🎉 Artifact processing complete!")