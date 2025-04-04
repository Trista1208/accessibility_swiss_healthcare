# Swiss Healthcare Website Accessibility Analysis

This repository contains a comprehensive analysis of the accessibility of Swiss healthcare websites based on Google Lighthouse audits.

## Project Overview

The analysis evaluates the accessibility compliance of 99 healthcare websites in Switzerland, focusing on key metrics including:
- Accessibility scores
- Performance metrics
- Best practices
- SEO standards
- Common accessibility issues


## Repository Structure

- **Data Files**:
  - `01_lighthouse_results_original.csv`: Raw data from Lighthouse audits
  - `02_lighthouse_data_cleaned.csv`: Cleaned and optimized data for analysis

- **Code**:
  - `01_data_collection.py`: Script for collecting the Lighthouse data
  - `02_clean_lighthouse_data.py`: Script for cleaning and optimizing the data
  - `03_analytics_and_eda.py`: Script for generating insights and visualizations

- **Results**:
  - `insights_summary.txt`: Comprehensive report with key findings and recommendations
  - `final_accessibility_analysis/`: Directory containing all visualizations and insights

## Key Findings

- Only 48.9% of analyzed healthcare websites pass accessibility standards (score ≥90)
- The average accessibility score is 88.1/100
- Websites have an average of 3.2 accessibility issues per site (based on sites with available issue data)
- There is a 41.0-point difference between the best and worst performing domains
- High Severity Issues, found on 92 sites
- Performance metrics are particularly problematic, with only 6.5% of sites passing standards

## Visualizations

The analysis includes several visualizations with a consistent red and dark blue color scheme:
- Accessibility score distributions
- Pass/fail breakdown
- Common accessibility issues
- Score correlations
- Severity distribution
- Comprehensive accessibility dashboard

## Recommendations

Based on the analysis, key recommendations include:
1. Prioritizing critical accessibility issues
2. Implementing regular accessibility audits
3. Adopting WCAG 2.1 AA standards
4. Providing accessibility training for development teams
5. Optimizing performance through better JavaScript loading and caching policies

## Usage

To run the analysis yourself:

```bash
# Clean and optimize the data
python 02_clean_lighthouse_data.py

# Generate insights and visualizations
python 03_analytics_and_eda.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions and issues please contact: jiaqi.yu@stud.hslu.ch
