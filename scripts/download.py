import os
import shutil
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from urllib.parse import urlparse
from enum import Enum
from rich.progress import Progress, BarColumn, TextColumn, ProgressColumn
from rich.console import Console
from rich.text import Text
from rich.style import Style
from agentsearch.dataset.papers import Paper
from agentsearch.apis import s2

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


class SegmentedBarColumn(ProgressColumn):
    """A progress bar column that shows different colored segments for successes and failures."""
    
    def __init__(self, bar_width=40):
        self.bar_width = bar_width
        super().__init__()
    
    def render(self, task) -> Text:
        """Render the segmented progress bar."""
        successes = task.fields.get('successes', 0)
        failures = task.fields.get('failures', 0)
        total_processed = successes + failures
        
        if total_processed == 0:
            return Text("█" * self.bar_width, style="dim")
        
        # Calculate proportions
        success_ratio = successes / total_processed
        failure_ratio = failures / total_processed
        
        # Calculate segment widths
        success_width = int(success_ratio * self.bar_width)
        failure_width = int(failure_ratio * self.bar_width)
        
        # Ensure we fill the entire bar width
        remaining = self.bar_width - success_width - failure_width
        if remaining > 0:
            if success_ratio >= failure_ratio:
                success_width += remaining
            else:
                failure_width += remaining
        
        # Create the segmented bar
        bar = Text()
        bar.append("█" * success_width, style="green")
        bar.append("█" * failure_width, style="red")
        
        return bar

def attempt_pdf_download(url: str, export_path: str) -> bool:
    resp = requests.get(url, allow_redirects=True, headers=headers, timeout=30)
                
    if not resp.status_code == 200:
        print(f"failed to download: url={url} status={resp.status_code}")
        return False

    if not resp.headers['Content-Type'].startswith('application/'): 
        print(f"unexpected content type: url={url} content-type={resp.headers['Content-Type']}")
        return False
    
    with open(export_path, 'wb') as f:
        f.write(resp.content)
    
    return True

def scrape_pdf_url(html: str, current_url: str) -> str | None:
    soup = BeautifulSoup(html, 'html.parser')
    pdf_meta = soup.find('meta', attrs={'name': 'citation_pdf_url'})
    if not pdf_meta:
        return None

    url = pdf_meta.get('content')
    if not url:
        return None
    
    # convert relative URL to absolute URL
    if url.startswith('/'):
        parsed = urlparse(current_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        url = base_url + url
    elif url.startswith('www.'):
        url = 'https://' + url
    
    return url

def download_paper(agent_id: str, s2_paper: s2.Paper) -> bool:
    """
    Downloads a paper through external IDs.
    Returns True if successful, False otherwise.
    """
    s2_id, external_ids = s2_paper
    export_path = f'papers/pdf/{agent_id}/{s2_id}.pdf'

    try:
        if 'ArXiv' in external_ids:
            arxiv_id = external_ids['ArXiv']
            url = f"https://arxiv.org/pdf/{arxiv_id}"
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                raise Exception(f"ArXiv download failed with status {response.status_code} for {arxiv_id}")
            if response.headers['Content-Type'] == 'application/pdf':
                with open(export_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                print(f"ArXiv response was not a PDF: content-type={response.headers['Content-Type']}")
            
        if 'DOI' in external_ids:
            doi = external_ids['DOI']
            
            unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email=m.gregoriadis@tudelft.nl"
            response = requests.get(unpaywall_url, timeout=30)
            if response.status_code != 200:
                print(f"Unpaywall download failed with status {response.status_code}")
                return False
            data = response.json()

            if not data.get('is_oa'):
                resp = requests.get(data['doi_url'], allow_redirects=True, headers=headers, timeout=30)
                
                if not resp.status_code == 200:
                    print(f"failed to download: url={resp.url} status={resp.status_code}")
                    return False

                if resp.headers['Content-Type'].startswith('application/'):
                    with open(export_path, 'wb') as f:
                        f.write(resp.content)
                    return True
                
                pdf_url = scrape_pdf_url(resp.text, resp.url)
                if not pdf_url:
                    return False
                if attempt_pdf_download(pdf_url, export_path):
                    return True
                
                return False

            for loc in data['oa_locations']:
                url = loc['url_for_pdf'] if loc.get('url_for_pdf') else loc['url']
                resp = requests.get(url, allow_redirects=True, headers=headers, timeout=30)
                
                if not resp.status_code == 200:
                    print(f"failed to download: url={resp.url} status={resp.status_code}")
                    continue

                if resp.headers['Content-Type'].startswith('application/'):
                    with open(export_path, 'wb') as f:
                        f.write(resp.content)
                    return True

                pdf_url = scrape_pdf_url(resp.text, resp.url)
                if not pdf_url:
                    return False
                if attempt_pdf_download(pdf_url, export_path):
                    return True
            
    except Exception as e:
        print(f"Error downloading paper {s2_id}: {e}")

    return False

if __name__ == "__main__":
    agents_df = pd.read_csv('data/authors.csv', index_col=0)
    
    # Segmented progress bar for success/failure tracking
    progress = Progress(
        TextColumn("[bold blue]Downloading Papers:", justify="right"),
        SegmentedBarColumn(bar_width=40),
        TextColumn("[green]✓ {task.fields[successes]}", justify="right"),
        TextColumn("[red]✗ {task.fields[failures]}", justify="right"),
        "•",
        TextColumn("[cyan]Processed: {task.completed}", justify="right"),
        console=Console(),
    )
    
    with progress:
        download_task = progress.add_task(
            "Processing papers", 
            total=None,  # No known total - will grow dynamically
            successes=0,
            failures=0
        )
        
        successes = 0
        failures = 0
        total_processed = 0

        for agent_id, agent_row in agents_df.iterrows():
            # Sleep to avoid rate limiting
            time.sleep(60 if total_processed > 0 and total_processed % 100 == 0 else 1)
            
            # Get S2 profile and papers for this agent
            try:
                s2_profile_id = s2.match_profile(agent_row)
                s2_papers = s2.get_papers(s2_profile_id)
            except Exception as e:
                print(f"Error processing agent {agent_row['name']}: {e}")
                continue
            
            if not s2_papers:
                print(f"No papers found for agent {agent_row['name']}")
                continue
                
            print(f"Found {len(s2_papers)} papers for {agent_row['name']}")

            agent_paper_dir = f'papers/pdf/{agent_id}'
            try:
                os.makedirs(agent_paper_dir)
            except FileExistsError:
                print(f"Agent {agent_id} already has papers")
                continue

            # Process papers sequentially
            for s2_paper in s2_papers:
                paper = Paper(s2_paper[0], agent_id)
                if paper.exists():
                    print(f"Paper {paper.id} already exists")
                    continue

                try:
                    if download_paper(agent_id, s2_paper):
                        successes += 1
                        print("✅", s2_paper)
                    else:
                        failures += 1
                        print("❌", s2_paper)
                except Exception as e:
                    print(f"Error processing paper: {e}")
                    failures += 1
                finally:
                    total_processed += 1
                    progress.update(
                        download_task,
                        completed=total_processed,
                        successes=successes,
                        failures=failures
                    )
            # If no papers downloaded, remove the agent directory
            if not os.listdir(agent_paper_dir):
                shutil.rmtree(agent_paper_dir)
                print(f"Removed empty agent directory {agent_paper_dir}")
        
        # Print final summary
        console = Console()
        console.print(f"\n{'='*60}")
        console.print("[bold green]Download Summary[/bold green]")
        console.print(f"[green]✓ Successful: {successes} ({successes/total_processed*100:.1f}%)[/green]")
        console.print(f"[red]✗ Failed: {failures} ({failures/total_processed*100:.1f}%)[/red]")
        console.print(f"[cyan]Total Processed: {total_processed}[/cyan]")
        console.print(f"{'='*60}")