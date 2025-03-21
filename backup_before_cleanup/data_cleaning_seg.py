import pandas as pd
import ast
from collections import defaultdict
import re
from datetime import datetime

def clean_data(df):
    # Remove any duplicate URLs
    df = df.drop_duplicates(subset=['URL'], keep='first')
    
    # Convert scores to numeric, replacing 'N/A' with NaN
    score_columns = ['Performance', 'Accessibility', 'Best Practices', 'SEO']
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean URL column
    df['URL'] = df['URL'].str.strip()
    
    return df

def get_wcag_guideline(title):
    """Map issues to WCAG guidelines."""
    title_lower = title.lower()
    
    wcag_mapping = {
        'contrast': '1.4.3 Contrast (Minimum)',
        'color': '1.4.1 Use of Color',
        'aria': '4.1.2 Name, Role, Value',
        'landmark': '1.3.1 Info and Relationships',
        'heading': '2.4.6 Headings and Labels',
        'alt': '1.1.1 Non-text Content',
        'keyboard': '2.1.1 Keyboard',
        'focus': '2.4.7 Focus Visible',
        'language': '3.1.1 Language of Page',
        'link': '2.4.4 Link Purpose',
        'button': '4.1.2 Name, Role, Value',
        'form': '3.3.2 Labels or Instructions'
    }
    
    for key, guideline in wcag_mapping.items():
        if key in title_lower:
            return guideline
    return 'Other WCAG Guidelines'

def determine_impact(title, severity):
    """Determine the impact level of an issue."""
    title_lower = title.lower()
    
    # Critical functionality impact
    if any(word in title_lower for word in ['button', 'form', 'submit', 'input', 'navigation']):
        return 'Critical - Affects Core Functionality'
    
    # User experience impact
    if any(word in title_lower for word in ['keyboard', 'focus', 'tab', 'aria']):
        return 'High - Affects User Experience'
    
    # Content understanding impact
    if any(word in title_lower for word in ['heading', 'alt', 'language', 'contrast']):
        return 'Medium - Affects Content Understanding'
    
    # Visual/aesthetic impact
    if any(word in title_lower for word in ['color', 'background']):
        return 'Low - Affects Visual Presentation'
    
    return 'Unknown Impact'

def get_priority_level(severity, impact):
    """Determine priority level based on severity and impact."""
    if severity == 'High' and 'Critical' in impact:
        return 'P0 - Immediate Action Required'
    elif severity == 'High' and 'High' in impact:
        return 'P1 - High Priority'
    elif severity == 'Medium' or ('High' in severity and 'Medium' in impact):
        return 'P2 - Medium Priority'
    elif severity == 'Low':
        return 'P3 - Low Priority'
    return 'P4 - To Be Evaluated'

def get_detailed_wcag_explanation(guideline_id):
    """Provide detailed explanations for WCAG guidelines with examples and technical requirements."""
    explanations = {
        '1.4.3': {
            'title': 'Contrast (Minimum)',
            'description': 'The visual presentation of text and images of text has a contrast ratio of at least 4.5:1.',
            'technical_requirements': [
                'Normal text: minimum contrast ratio of 4.5:1',
                'Large text (18pt or 14pt bold): minimum contrast ratio of 3:1',
                'Incidental text and logotypes: no contrast requirement'
            ],
            'examples': [
                'Black text (#000000) on white background (#FFFFFF) - ratio: 21:1',
                'Dark gray text (#595959) on light gray (#FFFFFF) - ratio: 7:1',
                'Brown text (#8B4513) on beige (#F5F5DC) - ratio: 4.5:1'
            ],
            'testing_tools': [
                'WebAIM Contrast Checker',
                'Chrome DevTools Color Picker',
                'WAVE Evaluation Tool'
            ]
        },
        '1.4.1': {
            'title': 'Use of Color',
            'description': 'Color is not used as the only visual means of conveying information, indicating an action, prompting a response, or distinguishing a visual element.',
            'technical_requirements': [
                'Provide text alternatives for color-based information',
                'Use patterns, shapes, or text in addition to color',
                'Ensure colorblind users can distinguish important information'
            ],
            'examples': [
                'Required form fields marked with both red color AND an asterisk (*)',
                'Links underlined in addition to being colored',
                'Error messages showing both red color AND an error icon'
            ],
            'testing_tools': [
                'Chrome DevTools Color Vision Deficiency Simulator',
                'Color Contrast Analyzer',
                'NoCoffee Vision Simulator'
            ]
        },
        '4.1.2': {
            'title': 'Name, Role, Value',
            'description': 'For all user interface components, the name, role, and value can be programmatically determined.',
            'technical_requirements': [
                'Use proper HTML elements with semantic meaning',
                'Provide labels for all form controls',
                'Implement ARIA labels and roles correctly',
                'Ensure custom controls have proper roles and states'
            ],
            'examples': [
                '<button aria-label="Close dialog">×</button>',
                '<input type="checkbox" aria-checked="true">',
                '<div role="tabpanel" aria-labelledby="tab1">'
            ],
            'testing_tools': [
                'WAVE Accessibility Tool',
                'aXe DevTools',
                'Chrome Accessibility Inspector'
            ]
        },
        '2.4.6': {
            'title': 'Headings and Labels',
            'description': 'Headings and labels describe topic or purpose clearly and are properly structured.',
            'technical_requirements': [
                'Use descriptive headings that outline the page structure',
                'Maintain proper heading hierarchy (h1-h6)',
                'Provide clear, descriptive labels for form controls',
                'Avoid duplicate headings unless in different sections'
            ],
            'examples': [
                '<h1>Main Article Title</h1>',
                '<h2>Section Overview</h2>',
                '<label for="email">Email Address:</label>'
            ],
            'testing_tools': [
                'HeadingsMap browser extension',
                'WAVE Outline View',
                'Screen reader testing'
            ]
        }
        # ... Add more guidelines as needed
    }
    
    # Extract the guideline number from the full guideline string
    guideline_num = guideline_id.split()[0]
    base_num = '.'.join(guideline_num.split('.')[:2])
    
    return explanations.get(base_num, {
        'title': 'Generic Guideline',
        'description': 'Ensure compliance with WCAG 2.1 guidelines.',
        'technical_requirements': ['Follow WCAG 2.1 specifications'],
        'examples': ['Refer to WCAG documentation'],
        'testing_tools': ['Automated accessibility testing tools']
    })

def generate_detailed_recommendation(issue):
    """Generate comprehensive recommendations with examples and technical details."""
    title_lower = issue['title'].lower()
    wcag_guideline = issue['wcag_guideline'].split()[0]
    explanation = get_detailed_wcag_explanation(wcag_guideline)
    
    base_recommendations = {
        'contrast': {
            'immediate_action': 'Use a color contrast checker to verify all text meets WCAG 2.1 requirements.',
            'technical_steps': [
                'Measure the contrast ratio between text and background',
                'Adjust colors to meet minimum requirements (4.5:1 for normal text, 3:1 for large text)',
                'Consider using darker shades for text or lighter backgrounds'
            ],
            'code_example': '''
// Example CSS with good contrast
.text-content {
    color: #333333; /* Dark gray text */
    background-color: #FFFFFF; /* White background */
}
'''
        },
        'aria': {
            'immediate_action': 'Review and implement proper ARIA attributes for all interactive elements.',
            'technical_steps': [
                'Add appropriate ARIA roles to custom elements',
                'Ensure all interactive elements have accessible names',
                'Implement proper ARIA states and properties'
            ],
            'code_example': '''
<!-- Example of proper ARIA usage -->
<button aria-expanded="false" aria-controls="menu-content">
    <span class="icon" aria-hidden="true"></span>
    <span>Menu</span>
</button>
'''
        },
        'keyboard': {
            'immediate_action': 'Ensure all interactive elements are keyboard accessible.',
            'technical_steps': [
                'Add visible focus indicators',
                'Implement proper tab order',
                'Ensure all actions can be performed with keyboard'
            ],
            'code_example': '''
/* Example CSS for keyboard accessibility */
:focus {
    outline: 2px solid #4A90E2;
    outline-offset: 2px;
}
'''
        }
        # ... Add more detailed recommendations
    }
    
    # Find matching recommendation
    for key, details in base_recommendations.items():
        if key in title_lower:
            return {
                'recommendation': details['immediate_action'],
                'technical_steps': details['technical_steps'],
                'code_example': details['code_example'],
                'wcag_details': explanation
            }
    
    return {
        'recommendation': 'Review and implement appropriate accessibility solutions.',
        'technical_steps': ['Consult WCAG 2.1 guidelines'],
        'code_example': '// Implement according to WCAG specifications',
        'wcag_details': explanation
    }

def parse_issue(issue_text):
    """Parse a single issue into its components with detailed recommendations."""
    if not issue_text.strip():
        return None
        
    # Extract components using regex
    pattern = r"(.*?)\s*\(Score:\s*([\d.]+|N/A)\)"
    match = re.match(pattern, issue_text)
    
    if not match:
        return None
        
    title = match.group(1).strip()
    score = match.group(2)
    
    # Determine severity based on score
    try:
        score_num = float(score) if score != 'N/A' else 0
        if score_num >= 0.8:
            severity = 'Low'
        elif score_num >= 0.5:
            severity = 'Medium'
        else:
            severity = 'High'
    except:
        severity = 'Unknown'
    
    # Get additional metadata
    wcag = get_wcag_guideline(title)
    impact = determine_impact(title, severity)
    priority = get_priority_level(severity, impact)
    detailed_rec = generate_detailed_recommendation({'title': title, 'severity': severity, 'wcag_guideline': wcag})
    
    return {
        'title': title,
        'score': score,
        'severity': severity,
        'wcag_guideline': wcag,
        'impact': impact,
        'priority': priority,
        'detailed_recommendation': detailed_rec['recommendation'],
        'technical_steps': detailed_rec['technical_steps'],
        'code_example': detailed_rec['code_example'],
        'wcag_details': detailed_rec['wcag_details']
    }

def categorize_issue(title):
    """Categorize an issue based on its title and return detailed categorization."""
    title_lower = title.lower()
    
    # Main category
    if any(word in title_lower for word in ['color', 'contrast', 'background']):
        main_category = 'color_contrast'
        sub_category = 'visual_design'
    elif 'aria' in title_lower:
        main_category = 'aria'
        if 'role' in title_lower:
            sub_category = 'roles'
        elif 'label' in title_lower:
            sub_category = 'labels'
        else:
            sub_category = 'general'
    elif any(word in title_lower for word in ['navigation', 'menu', 'landmark']):
        main_category = 'navigation'
        if 'landmark' in title_lower:
            sub_category = 'landmarks'
        elif 'menu' in title_lower:
            sub_category = 'menus'
        else:
            sub_category = 'general'
    elif any(word in title_lower for word in ['form', 'input', 'label', 'button']):
        main_category = 'forms'
        if 'label' in title_lower:
            sub_category = 'labels'
        elif 'input' in title_lower:
            sub_category = 'inputs'
        elif 'button' in title_lower:
            sub_category = 'buttons'
        else:
            sub_category = 'general'
    elif any(word in title_lower for word in ['image', 'img', 'alt']):
        main_category = 'images'
        if 'alt' in title_lower:
            sub_category = 'alt_text'
        else:
            sub_category = 'general'
    elif any(word in title_lower for word in ['link', 'href']):
        main_category = 'links'
        if 'name' in title_lower:
            sub_category = 'names'
        elif 'text' in title_lower:
            sub_category = 'text'
        else:
            sub_category = 'general'
    elif any(word in title_lower for word in ['heading', 'h1', 'h2', 'h3']):
        main_category = 'headings'
        if 'order' in title_lower:
            sub_category = 'hierarchy'
        else:
            sub_category = 'general'
    elif any(word in title_lower for word in ['keyboard', 'focus', 'tab']):
        main_category = 'keyboard'
        if 'focus' in title_lower:
            sub_category = 'focus'
        elif 'tab' in title_lower:
            sub_category = 'tabbing'
        else:
            sub_category = 'general'
    elif any(word in title_lower for word in ['language', 'lang']):
        main_category = 'language'
        sub_category = 'general'
    else:
        main_category = 'other'
        sub_category = 'general'
    
    return main_category, sub_category

def segment_accessibility_issues(issues_text):
    """Enhanced segmentation of accessibility issues with detailed categorization."""
    # Initialize categories with subcategories
    categories = defaultdict(lambda: defaultdict(list))
    
    if pd.isna(issues_text) or issues_text == '':
        return categories
    
    # Split issues by semicolon
    issues = issues_text.split(';')
    
    for issue in issues:
        parsed_issue = parse_issue(issue)
        if not parsed_issue:
            continue
            
        main_category, sub_category = categorize_issue(parsed_issue['title'])
        categories[main_category][sub_category].append(parsed_issue)
    
    return categories

def format_detailed_issues(issues_dict):
    """Format issues with comprehensive details and recommendations."""
    formatted = defaultdict(lambda: defaultdict(str))
    
    for main_category, subcategories in issues_dict.items():
        for sub_category, issues in subcategories.items():
            base_key = f"{main_category}_{sub_category}"
            
            # Format different aspects of the issues
            issues_text = []
            wcag_guidelines = set()
            priorities = []
            recommendations = []
            technical_details = []
            code_examples = []
            
            for issue in issues:
                # Basic issue information
                issues_text.append(f"{issue['title']} (Score: {issue['score']}, Severity: {issue['severity']})")
                
                # Collect metadata
                wcag_guidelines.add(issue['wcag_guideline'])
                priorities.append(issue['priority'])
                
                # Collect detailed recommendations
                recommendations.append(issue['detailed_recommendation'])
                technical_details.extend(issue['technical_steps'])
                if issue['code_example'].strip():
                    code_examples.append(issue['code_example'])
            
            # Store different aspects in the formatted dictionary
            formatted[f"{base_key}_details"] = '; '.join(issues_text)
            formatted[f"{base_key}_wcag"] = '; '.join(sorted(wcag_guidelines))
            formatted[f"{base_key}_priority"] = '; '.join(sorted(set(priorities)))
            formatted[f"{base_key}_recommendations"] = '; '.join(sorted(set(recommendations)))
            formatted[f"{base_key}_technical_steps"] = '; '.join(sorted(set(technical_details)))
            formatted[f"{base_key}_code_examples"] = '\n'.join(code_examples)
    
    return formatted

def main():
    # Read the CSV file
    print("Reading lighthouse results...")
    df = pd.read_csv('lighthouse_results.csv')
    
    # Clean the data
    print("Cleaning data...")
    df = clean_data(df)
    
    # Process accessibility issues
    print("Processing detailed accessibility issues...")
    
    # Create columns for detailed analysis
    all_categories = set()
    processed_rows = []
    
    # First pass to collect all possible category combinations
    for idx, row in df.iterrows():
        issues_dict = segment_accessibility_issues(row['Accessibility Issues'])
        formatted = format_detailed_issues(issues_dict)
        all_categories.update(formatted.keys())
        processed_rows.append(formatted)
    
    # Initialize all columns
    for category in all_categories:
        df[f'Accessibility_{category}'] = ''
    
    # Second pass to fill in the data
    for idx, formatted in enumerate(processed_rows):
        for category in all_categories:
            df.at[idx, f'Accessibility_{category}'] = formatted.get(category, '')
    
    # Add summary columns
    df['Total_Issues'] = df['Accessibility Issues'].apply(lambda x: len(x.split(';')) if pd.notna(x) and x != '' else 0)
    df['High_Severity_Issues'] = df.filter(like='Accessibility_').apply(
        lambda x: sum('Severity: High' in str(val) for val in x), axis=1
    )
    df['P0_Issues'] = df.filter(like='_priority').apply(
        lambda x: sum('P0' in str(val) for val in x), axis=1
    )
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'lighthouse_results_detailed_{timestamp}.csv'
    
    # Save the processed data
    print(f"Saving detailed results to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Generate summary statistics
    print("\nAnalysis Summary:")
    print(f"Total URLs processed: {len(df)}")
    print(f"Total categories identified: {len(all_categories)}")
    
    print("\nSeverity Distribution:")
    severity_counts = {
        'High': df['High_Severity_Issues'].sum(),
        'P0': df['P0_Issues'].sum(),
        'Total': df['Total_Issues'].sum()
    }
    for severity, count in severity_counts.items():
        print(f"- {severity}: {count}")
    
    print("\nAverage issues per category:")
    for category in sorted(all_categories):
        if 'details' in category:  # Only show main categories
            avg_issues = df[f'Accessibility_{category}'].apply(
                lambda x: len(x.split(';')) if pd.notna(x) and x != '' else 0
            ).mean()
            print(f"- {category}: {avg_issues:.2f}")
    
    # Generate WCAG compliance summary
    wcag_columns = [col for col in df.columns if '_wcag' in col]
    all_wcag = set()
    for col in wcag_columns:
        all_wcag.update([
            guideline.strip() 
            for guidelines in df[col].dropna() 
            for guideline in guidelines.split(';')
        ])
    
    print("\nWCAG Guidelines Coverage:")
    for guideline in sorted(all_wcag):
        print(f"- {guideline}")
    
    print("\nData processing completed!")

if __name__ == "__main__":
    main()
