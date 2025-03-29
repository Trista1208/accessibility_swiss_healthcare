"""
Healthcare-Specific Accessibility Analysis
---------------------------------------
This script performs specialized accessibility analysis for healthcare websites,
focusing on medical terminology, emergency information, appointment systems,
and medical document readability.

Author: Jiaqi Yu
Date: 2025-03-11
"""

import pandas as pd
import re
from collections import defaultdict
import numpy as np
from datetime import datetime
import glob
import os

def analyze_medical_terminology(content, issues):
    """
    Analyze accessibility of medical terminology.
    Checks for:
    - Medical term definitions/explanations
    - Presence of plain language alternatives
    - Medical abbreviation explanations
    """
    terminology_issues = []
    
    # Check for unexplained medical terms
    medical_terms_pattern = r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:syndrome|disease|disorder|condition|procedure|therapy))\b'
    found_terms = re.findall(medical_terms_pattern, content)
    
    # Check for unexplained abbreviations
    medical_abbrev_pattern = r'\b[A-Z]{2,}\b'
    abbreviations = re.findall(medical_abbrev_pattern, content)
    
    if found_terms:
        terminology_issues.append({
            'category': 'medical_terminology',
            'subcategory': 'complex_terms',
            'title': 'Complex Medical Terms Without Plain Language Alternatives',
            'severity': 'High',
            'terms': found_terms
        })
    
    if abbreviations:
        terminology_issues.append({
            'category': 'medical_terminology',
            'subcategory': 'abbreviations',
            'title': 'Medical Abbreviations Without Explanations',
            'severity': 'High',
            'terms': abbreviations
        })
    
    return terminology_issues

def analyze_emergency_info(content, issues):
    """
    Analyze accessibility of emergency information.
    Checks for:
    - Emergency contact visibility
    - Emergency procedure clarity
    - Quick access to critical information
    """
    emergency_issues = []
    
    # Check for emergency information placement
    emergency_keywords = ['emergency', 'urgent', 'crisis', 'immediate', '911']
    emergency_content = any(keyword in content.lower() for keyword in emergency_keywords)
    
    if emergency_content:
        # Check if emergency info is prominently placed
        if not any('landmark' in str(issue) and 'main' in str(issue) for issue in issues):
            emergency_issues.append({
                'category': 'emergency_information',
                'subcategory': 'placement',
                'title': 'Emergency Information Not Prominently Placed',
                'severity': 'Critical',
                'recommendation': 'Place emergency information in a prominent, easily accessible location'
            })
        
        # Check for keyboard accessibility
        if any('keyboard' in str(issue) for issue in issues):
            emergency_issues.append({
                'category': 'emergency_information',
                'subcategory': 'accessibility',
                'title': 'Emergency Information Not Keyboard Accessible',
                'severity': 'Critical',
                'recommendation': 'Ensure emergency information is accessible via keyboard navigation'
            })
    
    return emergency_issues

def analyze_appointment_system(content, issues):
    """
    Analyze accessibility of appointment booking systems.
    Checks for:
    - Form accessibility
    - Clear instructions
    - Error handling
    - Time selection accessibility
    """
    appointment_issues = []
    
    # Check for appointment-related content
    appointment_keywords = ['appointment', 'schedule', 'booking', 'reservation']
    has_appointment_system = any(keyword in content.lower() for keyword in appointment_keywords)
    
    if has_appointment_system:
        # Check form accessibility
        form_issues = [issue for issue in issues if 'form' in str(issue).lower()]
        if form_issues:
            appointment_issues.append({
                'category': 'appointment_system',
                'subcategory': 'form_accessibility',
                'title': 'Appointment Form Accessibility Issues',
                'severity': 'High',
                'related_issues': form_issues
            })
        
        # Check for ARIA labels in date/time selection
        if any('aria' in str(issue) for issue in issues):
            appointment_issues.append({
                'category': 'appointment_system',
                'subcategory': 'datetime_accessibility',
                'title': 'Date/Time Selection Not Fully Accessible',
                'severity': 'High',
                'recommendation': 'Implement proper ARIA labels for date and time selection components'
            })
    
    return appointment_issues

def analyze_document_readability(content):
    """
    Analyze readability of medical documents.
    Checks for:
    - Text complexity
    - Reading level
    - Document structure
    """
    readability_issues = []
    
    # Check sentence length (rough complexity indicator)
    sentences = re.split(r'[.!?]+', content)
    long_sentences = [s for s in sentences if len(s.split()) > 20]
    
    if long_sentences:
        readability_issues.append({
            'category': 'document_readability',
            'subcategory': 'complexity',
            'title': 'Complex Sentence Structure in Medical Content',
            'severity': 'Medium',
            'recommendation': 'Break down complex sentences for better readability'
        })
    
    # Check for proper document structure
    if not re.search(r'<h1.*?>.*?</h1>', content, re.I):
        readability_issues.append({
            'category': 'document_readability',
            'subcategory': 'structure',
            'title': 'Missing Main Heading in Medical Document',
            'severity': 'High',
            'recommendation': 'Add clear main heading to improve document structure'
        })
    
    return readability_issues

def generate_healthcare_report(df):
    """Generate a comprehensive healthcare accessibility report."""
    report = {
        'total_sites': len(df),
        'medical_terminology': {'total': 0, 'percentage': 0},
        'emergency_information': {'total': 0, 'percentage': 0},
        'appointment_system': {'total': 0, 'percentage': 0},
        'document_readability': {'total': 0, 'percentage': 0}
    }
    
    # Analyze each category
    for category in report.keys():
        if category == 'total_sites':
            continue
            
        col = f'Healthcare_{category}_issues'
        if col in df.columns:
            issues = df[col].notna().sum()
            report[category]['total'] = issues
            report[category]['percentage'] = (issues / len(df)) * 100
    
    return report

def print_detailed_report(df, report):
    """Print a detailed analysis report with statistics and recommendations."""
    print("\n=== Healthcare Accessibility Analysis Report ===")
    print("\n1. Overall Statistics:")
    print(f"Total websites analyzed: {report['total_sites']}")
    
    # Calculate overall statistics
    avg_accessibility = df['Accessibility'].mean()
    med_accessibility = df['Accessibility'].median()
    print(f"Average Accessibility Score: {avg_accessibility:.1f}")
    print(f"Median Accessibility Score: {med_accessibility:.1f}")
    
    print("\n2. Healthcare-Specific Issues:")
    for category in ['medical_terminology', 'emergency_information', 'appointment_system', 'document_readability']:
        print(f"\n{category.replace('_', ' ').title()}:")
        print(f"- Affected Sites: {report[category]['total']} ({report[category]['percentage']:.1f}%)")
        
        # Get severity distribution
        severity_col = f'Healthcare_{category}_severity'
        if severity_col in df.columns:
            severities = df[severity_col].value_counts()
            if not severities.empty:
                print("- Severity Distribution:")
                for severity, count in severities.items():
                    if pd.notna(severity) and severity:
                        print(f"  * {severity}: {count} sites ({(count/len(df))*100:.1f}%)")
    
    print("\n3. Critical Issues:")
    critical_sites = df[df['Healthcare_emergency_information_severity'] == 'Critical']
    if not critical_sites.empty:
        print(f"\nSites with Critical Emergency Information Issues: {len(critical_sites)}")
        for _, site in critical_sites.head().iterrows():
            print(f"- {site['URL']}")
    
    print("\n4. Common Issues by Category:")
    for category in ['medical_terminology', 'emergency_information', 'appointment_system', 'document_readability']:
        issues_col = f'Healthcare_{category}_issues'
        if issues_col in df.columns:
            all_issues = []
            for issues in df[issues_col].dropna():
                if issues:
                    all_issues.extend([i.strip() for i in str(issues).split(';')])
            
            if all_issues:
                print(f"\n{category.replace('_', ' ').title()}:")
                from collections import Counter
                common_issues = Counter(all_issues).most_common(3)
                for issue, count in common_issues:
                    print(f"- {issue}: {count} occurrences")
    
    print("\n5. Recommendations:")
    print("\nMedical Terminology:")
    print("- Provide plain language alternatives for medical terms")
    print("- Include explanations for medical abbreviations")
    print("- Use tooltips or expandable definitions for complex terms")
    
    print("\nEmergency Information:")
    print("- Ensure emergency contact information is prominently displayed")
    print("- Make emergency information keyboard accessible")
    print("- Use clear visual indicators for urgent information")
    
    print("\nAppointment Systems:")
    print("- Implement fully accessible form controls")
    print("- Provide clear error messages and validation")
    print("- Ensure date/time selection is screen reader friendly")
    
    print("\nDocument Readability:")
    print("- Break down complex medical information into digestible sections")
    print("- Use clear headings and structure")
    print("- Provide printable versions of important documents")

def main():
    # Load the detailed accessibility results
    print("Loading accessibility data...")
    
    # Find the most recent detailed results file
    detailed_files = glob.glob('lighthouse_results_detailed_*.csv')
    if not detailed_files:
        print("Error: No detailed results file found. Please run the data_cleaning_seg.py script first.")
        return
    
    # Get the most recent file
    latest_file = max(detailed_files, key=os.path.getctime)
    print(f"Loading data from {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    print("\nAnalyzing healthcare-specific accessibility issues...")
    
    # Initialize new columns for healthcare-specific analysis
    healthcare_categories = [
        'medical_terminology', 'emergency_information',
        'appointment_system', 'document_readability'
    ]
    
    for category in healthcare_categories:
        df[f'Healthcare_{category}_issues'] = ''
        df[f'Healthcare_{category}_severity'] = ''
        df[f'Healthcare_{category}_recommendations'] = ''
    
    # Analyze each website
    for idx, row in df.iterrows():
        content = str(row['Accessibility Issues'])
        issues = content.split(';')
        
        # Perform healthcare-specific analyses
        terminology_issues = analyze_medical_terminology(content, issues)
        emergency_issues = analyze_emergency_info(content, issues)
        appointment_issues = analyze_appointment_system(content, issues)
        readability_issues = analyze_document_readability(content)
        
        # Update DataFrame with results
        df.at[idx, 'Healthcare_medical_terminology_issues'] = '; '.join(str(issue) for issue in terminology_issues)
        df.at[idx, 'Healthcare_emergency_information_issues'] = '; '.join(str(issue) for issue in emergency_issues)
        df.at[idx, 'Healthcare_appointment_system_issues'] = '; '.join(str(issue) for issue in appointment_issues)
        df.at[idx, 'Healthcare_document_readability_issues'] = '; '.join(str(issue) for issue in readability_issues)
    
    # Generate comprehensive report
    report = generate_healthcare_report(df)
    
    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'healthcare_accessibility_analysis_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    
    # Print detailed report
    print_detailed_report(df, report)
    
    print(f"\nDetailed results saved to: {output_file}")
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main() 