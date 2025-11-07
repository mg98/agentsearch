import csv

def merge_and_filter():
    """
    Merge agents.csv with results.csv and filter out unsuccessful scrapes.
    """
    results_map = {}

    with open('results.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results_map[row['id']] = row

    successful_records = []

    with open('agents.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scientist_id = row['Unnamed: 0']

            if scientist_id in results_map:
                result = results_map[scientist_id]

                if result['faculty'] and not result.get('error'):
                    merged = {
                        'id': scientist_id,
                        'name': row['name'],
                        'research_fields': row['research_fields'],
                        'citation_count': row['citation_count'],
                        'scholar_url': row['scholar_url'],
                        'faculty': result['faculty'],
                        'department': result['department'],
                        'group': result['group'],
                        'profile_url': result['url']
                    }
                    successful_records.append(merged)

    with open('scientists_complete.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'name', 'faculty', 'department', 'group',
                      'research_fields', 'citation_count', 'scholar_url', 'profile_url']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(successful_records)

    print("=" * 60)
    print("MERGE COMPLETE!")
    print("=" * 60)
    print(f"Successfully merged: {len(successful_records)} scientists")
    print(f"Filtered out: {1494 - len(successful_records)} scientists")
    print("=" * 60)
    print(f"\nOutput file: scientists_complete.csv")
    print("\nColumns:")
    print("  - id, name")
    print("  - faculty, department, group")
    print("  - research_fields, citation_count")
    print("  - scholar_url, profile_url")
    print("=" * 60)

if __name__ == '__main__':
    merge_and_filter()
