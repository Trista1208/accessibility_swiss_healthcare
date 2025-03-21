import requests
import csv
import subprocess
import json
import time

# Your Google Custom Search API endpoint
API_URL = "https://www.googleapis.com/customsearch/v1"
API_KEY = "AIzaSyAKV6pVP15EYTCR90RDQWE5US1XQfSdg_g"
CX = "07022b2a0931549f0"
QUERY = "healthcare hospital clinic Switzerland"
COUNTRY = "countryCH"

# Function to fetch top 100 search results
def fetch_search_results(query, api_key, cx, country, num_results=100):
    results = []
    for start in range(1, num_results + 1, 10):  # Fetch 10 results at a time
        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "cr": country,
            "num": 10,  # Max results per request
            "start": start,
        }
        response = requests.get(API_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            if "items" in data:
                results.extend(data["items"])
            else:
                print(f"No results found for start={start}")
        else:
            print(f"Error fetching results for start={start}: {response.status_code}")
        time.sleep(1)  # Avoid hitting API rate limits
    return results[:num_results]

# Function to run Lighthouse test and extract detailed feedback
def run_lighthouse(url):
    try:
        # Run Lighthouse test using Node.js
        lighthouse_path = "/usr/local/bin/lighthouse"  # Update this path if necessary
        command = f"{lighthouse_path} {url} --output=json --quiet --chrome-flags='--headless'"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            report = json.loads(result.stdout)
            
            # Extract scores
            scores = {
                "performance": report["categories"]["performance"]["score"] * 100,
                "accessibility": report["categories"]["accessibility"]["score"] * 100,
                "best_practices": report["categories"]["best-practices"]["score"] * 100,
                "seo": report["categories"]["seo"]["score"] * 100,
            }

            # Extract accessibility feedback
            accessibility_audits = report["audits"]
            accessibility_feedback = []
            for audit_id, audit_result in accessibility_audits.items():
                if audit_result["score"] != 1:  # Only include failed or non-perfect audits
                    accessibility_feedback.append({
                        "id": audit_id,
                        "title": audit_result["title"],
                        "description": audit_result["description"],
                        "score": audit_result["score"],
                        "details": audit_result.get("details", {}),
                    })

            return {
                "scores": scores,
                "accessibility_feedback": accessibility_feedback,
            }
        else:
            print(f"Lighthouse test failed for {url}: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error running Lighthouse for {url}: {e}")
        return None

# Function to save results to CSV
def save_to_csv(data, filename="lighthouse_results.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Rank", "Title", "URL", 
            "Performance", "Accessibility", "Best Practices", "SEO",
            "Accessibility Issues"
        ])
        for idx, item in enumerate(data, start=1):
            accessibility_issues = item.get("lighthouse", {}).get("accessibility_feedback", [])
            issues_summary = "; ".join([f"{issue['title']} (Score: {issue['score']})" for issue in accessibility_issues])
            
            writer.writerow([
                idx,
                item.get("title", "N/A"),
                item.get("link", "N/A"),
                item.get("lighthouse", {}).get("scores", {}).get("performance", "N/A"),
                item.get("lighthouse", {}).get("scores", {}).get("accessibility", "N/A"),
                item.get("lighthouse", {}).get("scores", {}).get("best_practices", "N/A"),
                item.get("lighthouse", {}).get("scores", {}).get("seo", "N/A"),
                issues_summary,
            ])
    print(f"Results saved to {filename}")

# Main script
if __name__ == "__main__":
    # Fetch top 100 search results
    print("Fetching search results...")
    search_results = fetch_search_results(QUERY, API_KEY, CX, COUNTRY)

    # Run Lighthouse tests for each URL
    print("Running Lighthouse tests...")
    for result in search_results:
        url = result.get("link")
        if url:
            print(f"Testing {url}...")
            lighthouse_result = run_lighthouse(url)
            if lighthouse_result:
                result["lighthouse"] = lighthouse_result
            else:
                result["lighthouse"] = {
                    "scores": {
                        "performance": "N/A",
                        "accessibility": "N/A",
                        "best_practices": "N/A",
                        "seo": "N/A",
                    },
                    "accessibility_feedback": [],
                }
            time.sleep(2)  # Add delay to avoid overloading the system
        else:
            print("Skipping invalid URL.")

    # Save results to CSV
    save_to_csv(search_results)