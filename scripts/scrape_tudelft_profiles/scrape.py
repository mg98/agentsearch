import csv
import os
import time
import requests
from bs4 import BeautifulSoup

API_KEY = os.environ.get('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = '75b4e204161404688'
RESULTS_FILE = 'results.csv'
INPUT_FILE = 'agents.csv'

def google_search(query):
    """
    Search using Google Custom Search API and return the first result URL.
    """
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': API_KEY,
        'cx': SEARCH_ENGINE_ID,
        'q': query,
        'num': 1
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'items' in data and len(data['items']) > 0:
            return data['items'][0]['link']
        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"  Rate limit hit. Please wait or enable billing.")
            raise
        print(f"  HTTP Error: {e}")
        return None
    except Exception as e:
        print(f"  Error searching: {e}")
        return None

def extract_profile_info(html_content):
    """
    Extract faculty, department, and group from TU Delft profile page HTML.
    Returns a tuple: (faculty, department, group)
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    lists = soup.find_all('ul', class_='list-reset')

    for ul in lists:
        items = ul.find_all('li', recursive=False)

        all_text = ' '.join([item.get_text() for item in items])

        if 'Pure profile' in all_text and len(items) in [3, 4]:
            links = [item.find('a') for item in items]

            if len(items) == 4:
                faculty = links[0].get_text().strip() if links[0] else ''
                department = links[1].get_text().strip() if links[1] else ''
                group = links[2].get_text().strip() if links[2] else ''
                return faculty, department, group
            elif len(items) == 3:
                faculty = links[0].get_text().strip() if links[0] else ''
                department = links[1].get_text().strip() if links[1] else ''
                return faculty, department, ''

    return '', '', ''

def scrape_scientist(scientist_id, name):
    """
    Search for scientist, fetch their profile, and extract organizational info.
    Returns a dict with id, faculty, department, and group.
    """
    print(f"Processing {scientist_id}: {name}")

    try:
        search_url = google_search(name)
    except requests.exceptions.HTTPError:
        return {
            'id': scientist_id,
            'faculty': '',
            'department': '',
            'group': '',
            'url': '',
            'error': 'rate_limit'
        }

    if not search_url:
        print(f"  No search results")
        return {
            'id': scientist_id,
            'faculty': '',
            'department': '',
            'group': '',
            'url': '',
            'error': 'no_results'
        }

    print(f"  URL: {search_url}")

    try:
        response = requests.get(search_url, timeout=15)
        response.raise_for_status()
        html_content = response.text

        faculty, department, group = extract_profile_info(html_content)

        print(f"  Faculty: {faculty}")
        print(f"  Department: {department}")
        print(f"  Group: {group}")

        return {
            'id': scientist_id,
            'faculty': faculty,
            'department': department,
            'group': group,
            'url': search_url,
            'error': ''
        }
    except Exception as e:
        print(f"  Error fetching: {e}")
        return {
            'id': scientist_id,
            'faculty': '',
            'department': '',
            'group': '',
            'url': search_url,
            'error': str(e)
        }

def get_processed_ids():
    """
    Get set of already processed scientist IDs from results file.
    """
    if not os.path.exists(RESULTS_FILE):
        return set()

    processed = set()
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row['id'])
    return processed

def append_result(result):
    """
    Append a single result to the CSV file.
    """
    file_exists = os.path.exists(RESULTS_FILE)

    with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'faculty', 'department', 'group', 'url', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

def main():
    if not API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set")
        return

    processed_ids = get_processed_ids()
    total = 0
    processed = len(processed_ids)
    skipped = 0

    print(f"Resuming scraper...")
    print(f"Already processed: {processed} scientists")
    print("=" * 60)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            total += 1
            scientist_id = row['Unnamed: 0']
            name = row['name']

            if scientist_id in processed_ids:
                skipped += 1
                continue

            try:
                result = scrape_scientist(scientist_id, name)
                append_result(result)

                if result.get('error') == 'rate_limit':
                    print("\n" + "=" * 60)
                    print("RATE LIMIT REACHED!")
                    print(f"Processed: {total - skipped} new scientists")
                    print(f"Total in file: {processed + (total - skipped)}")
                    print(f"Remaining: {1494 - (processed + (total - skipped))}")
                    print("=" * 60)
                    print("\nOptions:")
                    print("1. Enable billing in Google Cloud Console")
                    print("2. Wait until tomorrow for free quota to reset")
                    print("3. Run this script again to resume")
                    return

                time.sleep(1)

            except KeyboardInterrupt:
                print("\n\nInterrupted! Progress saved.")
                print(f"Processed: {total - skipped} scientists")
                return

    print("\n" + "=" * 60)
    print(f"Completed! Processed {total - skipped} new scientists")
    print(f"Total: {processed + (total - skipped)}/{total} scientists")
    print(f"Skipped (already done): {skipped}")
    print(f"Results saved to {RESULTS_FILE}")
    print("=" * 60)

if __name__ == '__main__':
    main()
