from playwright.sync_api import sync_playwright
from _puppet import ScraperConfig, EtfScraper

config = ScraperConfig(base_url="https://etfdb.com")

with sync_playwright() as pw: 
    browser = pw.chromium.launch(headless=config.headless)

    try: 
        sectors = ['technology', 'consumer-discretionaries', 
        'consumer-staples', 'materials', 'industrials', 'real-estate',
         'financials', 'healthcare', 'telecom', 'utilities', 'energy']
        result = []
        
        scraper = EtfScraper(browser, config)
        scraper.set_page_cookies()

        etfs = scraper.scrape_sectors(sectors)
        for etf in etfs:
            result.append(scraper.scrape_etf_holdings(etf))

        scraper.save_to_json(result)
        
    finally:
        scraper.close()