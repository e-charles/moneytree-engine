from dataclasses import dataclass
from playwright.sync_api import Browser, Page

"""
do not need to make a class but can.... 
need to lauch browser and get all the necesarry data from etf.db 
KEY: add data validation checks for when site changes
"""

@dataclass 
class ScraperConfig: 
    base_url: str
    headless: bool = True 
    timeout_ms: int = 30_000


class EtfScraper:
    def __init__(self, browser: Browser, config: ScraperConfig):
        self.browser = browser
        self.config = config
        self.context = browser.new_context()
        self.context.set_default_timeout(config.timeout_ms)
        self.cookies = None

    def set_page_cookies(self) -> None:
        page = self.context.new_page();

        try: 
            url = self.config.base_url
            page.goto(url, wait_until="domcontentloaded")

            # get cookie or fail clearly 
            cookie_dict = page.cookies()
            if cookie_dict.count() == 0:
                raise ValueError(
                    "Expected to get page cookies. None were found."
                )

            self.cookies = cookie_dict
            self.context.add_cookies(cookie_dict)

        finally: 
            page.close()

    def scrape_sectors(self, sectors: list[str] ) -> list[str]:
        """
        Gets the top 10 total assets ($MM) from a sector 
        list = ['technology', ...]
        """
        page = self.context.new_page()
        results = []

        try: 
            for sec in sectors:
                url = f"{self.config.base_url}/etfs/sector/{sec}"
                page.goto(url, wait_until="domcontentloaded")



                results.append([]) # set the sector and list of etf's 
                # want to have a delay before next page
            
            return results
        
        finally:
            page.close()




    def scrape_etf_holdings(self, symbol: str) -> list[str]:
        """
        From a etf symbol, function will get the top 15 holdings and validate they are true tickers
        """
        page = self.context.new_page()

        try: 
            url = f"{self.config.base_url}/etf/{symbol}/#holdings"
            page.goto(url, wait_until='domcontentloaded')

            # need to check for N/A symbols in holdings 

            return []

        finally: 
            page.close()

    def save_to_json(self, data: list[str]) -> None:
        return

    def close(self) -> None:
        self.context.close()