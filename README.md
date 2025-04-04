# Swiss Healthcare Website Accessibility Analysis

This repository contains a comprehensive analysis of the accessibility of Swiss healthcare websites based on Google Lighthouse audits conducted in March 2025.

## Project Overview

The analysis evaluates the accessibility compliance of 92 Swiss healthcare websites, focusing on:
- Accessibility scores and compliance rates
- Common accessibility issues and their severity
- Performance, best practices, and SEO metrics
- Correlations between accessibility issues and overall scores
- Recommendations for improvement

## Repository Structure

- **Data Collection and Processing**:
  - `01_data_collection.py`: Script for collecting Lighthouse audit data
  - `01_lighthouse_results_original.csv`: Raw data from initial audits
  - `02_clean_lighthouse_data.py`: Data cleaning and preprocessing script
  - `02_lighthouse_data_cleaned.csv`: Processed dataset for analysis
  - `03_analytics_and_eda.py`: Analysis and visualization generation script

- **Results and Insights**:
  - `insights_summary.txt`: Executive summary of findings and recommendations
  - `final_accessibility_analysis/`: Directory containing visualizations and detailed insights
  - `presentation for accessibility analysis.pages`: Presentation of key findings

## Key Findings

- **Accessibility Compliance**: Only 48.9% of healthcare websites pass accessibility standards (score ≥90/100)
- **Performance Metrics**: 
  - Average accessibility score: 88.1/100
  - Average number of accessibility issues: 3.2 per site
  - 41.0-point gap between highest and lowest performing sites
- **Common Issues**:
  - High severity issues present on 92 sites (100% of sample)
  - Incorrect ARIA attribute usage (38 sites)
  - Missing form labels (31 sites)
  - Non-keyboard-focusable interactive controls (29 sites)
- **Keyboard Accessibility**: Only 31.5% of sites were tested for keyboard focus accessibility

## Visualizations

The analysis includes a comprehensive set of visualizations with a consistent color scheme:
- Score distributions and thresholds (`score_distribution.png`)
- Pass/fail breakdown by category (`accessibility_pass_fail.png`)
- Common accessibility issues and their frequency (`specific_issues.png`)
- Issue severity distribution (`severity_distribution.png`)
- Performance metric comparisons (`metric_comparisons.png`)
- Issue correlations and score impact (`issue_correlations.png`, `score_impact.png`)
- Consolidated accessibility dashboard (`accessibility_dashboard.png`)

## Recommendations

Based on the analysis, key recommendations include:
1. **Address High-Priority Issues**:
   - Fix form accessibility with proper labels
   - Correct ARIA attribute implementation
   - Ensure keyboard navigation for all interactive elements

2. **Adopt Accessibility Standards**:
   - Implement WCAG 2.1 AA standards across all healthcare websites
   - Incorporate accessibility testing throughout the development lifecycle
   - Test with actual assistive technologies and users with disabilities

3. **Improve Performance**:
   - Optimize page load times and interactive elements
   - Implement better JavaScript loading and caching policies
   - Regularly audit and update accessibility features

## Usage

To reproduce this analysis:

```bash
# Clean and preprocess the raw data
python 02_clean_lighthouse_data.py

# Generate insights and visualizations
python 03_analytics_and_eda.py
```

## Contact

For questions or inquiries about this analysis, please contact: jiaqi.yu@stud.hslu.ch

Report Date: March 30, 2025
