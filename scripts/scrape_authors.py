from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from dataclasses import dataclass, asdict
import pandas as pd

GOOGLE_SCHOLAR_ORG_ID = "814705155794667179"  # TU Delft
base_url = f"https://scholar.google.com/citations?view_op=view_org&hl=en&org={GOOGLE_SCHOLAR_ORG_ID}"

@dataclass
class Author:
    name: str
    research_fields: list[str]
    scholar_url: str

def setup_driver():
    """Set up Chrome driver with options to avoid detection"""
    chrome_options = Options()
    
    # Add user agent to appear as a real browser
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Disable automation indicators
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Additional options to appear more like a real browser
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins-discovery")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Optional: Use existing Chrome profile (uncomment and modify path if needed)
    chrome_options.add_argument("--user-data-dir=/Users/marcel/Library/Application Support/Google/Chrome/Default")
    
    # Use webdriver-manager to automatically handle ChromeDriver version
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Execute script to remove webdriver property
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def extract_authors_from_page(driver) -> list[Author]:
    """Extract authors from the current page"""
    authors = []
    
    # Wait for the main content container to be present
    container = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "gsc_sa_ccl"))
    )
    
    # Find all author entries
    for entry in  container.find_elements(By.CLASS_NAME, "gsc_1usr"):
        name_element = entry.find_element(By.CLASS_NAME, "gs_ai_name").find_element(By.TAG_NAME, "a")
        author_name = name_element.text.replace('"', '').strip()
        author_href = name_element.get_attribute('href')
        
        interests_container = entry.find_element(By.CLASS_NAME, "gs_ai_int")
        interest_elements = interests_container.find_elements(By.CLASS_NAME, "gs_ai_one_int")
        interests = [interest.text.strip() for interest in interest_elements]
            
        authors.append(Author(author_name, interests, author_href))
    
    return authors

def get_next_page_url(driver):
    """Find the next page button and extract its onclick URL"""
    try:
        # Wait for pagination section to be present
        pagination = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "gsc_authors_bottom_pag"))
        )
        
        # Find next button within pagination
        next_button = pagination.find_element(By.CLASS_NAME, "gsc_pgn_pnx")
        
        # Get onclick attribute
        onclick_value = next_button.get_attribute("onclick")
        if onclick_value:
            relative_url = onclick_value.split("'")[1]
            relative_url = relative_url.encode('latin1').decode('unicode-escape') 
            return "https://scholar.google.com" + relative_url
        else:
            return None
            
    except TimeoutException:
        print("Timeout waiting for pagination section")
        return None
    except NoSuchElementException:
        print("Could not find next button")
        return None

if __name__ == "__main__":
    driver = setup_driver()
    driver.get(base_url)
    input("Please login manually and press Enter when ready...")
    all_authors: list[Author] = []
    next_url = base_url

    try:
        while next_url is not None:
            driver.get(next_url)
            authors = extract_authors_from_page(driver)
            all_authors.extend(authors)
            next_url = get_next_page_url(driver)
    finally:
        authors_dicts = list(map(asdict, all_authors))
        df = pd.DataFrame(authors_dicts)
        # Remove duplicates, keeping entry with most research fields for each name
        df = df.loc[df.groupby('name')['research_fields'].apply(lambda x: x.apply(len).idxmax())]
        df.to_csv('data/authors.csv', index=True)
        print(f"Saved {len(all_authors)} authors to data/authors.csv")
        print("Last page visited (failed): ", next_url)

        driver.quit()
