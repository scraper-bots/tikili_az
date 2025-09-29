import asyncio
import aiohttp
from bs4 import BeautifulSoup
import csv
import pandas as pd
from typing import List, Dict
import re
from urllib.parse import urljoin

class TikiliScraper:
    def __init__(self):
        self.base_url = "https://tikili.az"
        self.search_url = "https://tikili.az/elan-searchajax.php"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,ru;q=0.7,az;q=0.6',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': 'https://tikili.az/elan-search.php?dil=az&realtor=0&emlak_type=menzil&',
            'DNT': '1'
        }
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_listing_urls(self, start: int = 0) -> List[str]:
        """Fetch listing URLs from search API"""
        params = {
            'dil': 'az',
            'realtor': '0',
            'emlak_type': 'menzil',
            'start': start
        }

        try:
            async with self.session.get(self.search_url, params=params) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract listing URLs from <a> tags
                urls = []
                for link in soup.select('a[href^="elan-item.php"]'):
                    href = link.get('href')
                    full_url = urljoin(self.base_url, href)
                    urls.append(full_url)

                return urls
        except Exception as e:
            print(f"Error fetching listing URLs at start={start}: {e}")
            return []

    async def get_all_listing_urls(self, max_pages: int = 100) -> List[str]:
        """Get all listing URLs by paginating through results"""
        all_urls = []

        # Execute in batches to avoid overwhelming the server
        batch_size = 10
        for batch_num in range(0, max_pages, batch_size):
            # Create tasks for this batch only
            tasks = []
            for page in range(batch_num, min(batch_num + batch_size, max_pages)):
                start = page * 40
                tasks.append(self.get_listing_urls(start))

            results = await asyncio.gather(*tasks)

            empty_results = 0
            for urls in results:
                if not urls:
                    empty_results += 1
                else:
                    all_urls.extend(urls)

            # If we get mostly empty results, stop pagination
            if empty_results > len(results) // 2:
                print(f"No more listings found at page {batch_num // batch_size}")
                break

            print(f"Fetched {len(all_urls)} listing URLs so far...")
            await asyncio.sleep(1)  # Be polite to the server

        return all_urls

    async def scrape_listing_details(self, url: str) -> Dict:
        """Scrape detailed information from a listing page"""
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                data = {
                    'url': url,
                    'listing_id': re.search(r'elan=(\d+)', url).group(1) if re.search(r'elan=(\d+)', url) else None
                }

                # Extract property details from table
                details = {}
                for row in soup.select('table tr'):
                    cells = row.select('td')
                    if len(cells) >= 2:
                        label_elem = cells[0].select_one('b')
                        if label_elem:
                            label = label_elem.get_text(strip=True).rstrip(':')
                            value = cells[1].get_text(strip=True)
                            # Clean value (remove images, etc)
                            value = re.sub(r'\s+', ' ', value).strip()
                            if value and value != 'inf $':
                                details[label] = value

                # Extract main fields
                data['listing_number'] = details.get('Elan nömrəsi', None)
                data['posted_date'] = details.get('Yerləşdirilib', None)
                data['property_type'] = details.get('Elan növü', None)
                data['area'] = details.get('Sahəsi', None)
                data['rooms'] = details.get('Otaq sayı', None)
                data['floor'] = details.get('Mərtəbəsi', None)
                data['repair'] = details.get('Təmiri', None)
                data['document'] = details.get('Sənədi', None)
                data['price_azn'] = details.get('Qiyməti AZN-lə', None)
                data['price_usd'] = details.get('Qiyməti $-la', None)
                data['district'] = details.get('Rayon', None)
                data['area_location'] = details.get('Yerləşdiyi ərazi', None)
                data['address'] = details.get('Ünvan', None)
                data['metro'] = details.get('Yaxınlıqdakı metro', None)

                # Additional fields
                data['mtk'] = details.get('MTK', None)
                data['layout'] = details.get('Layihə', None)
                data['gas'] = details.get('Binada qaz', None)
                data['su'] = details.get('S/U', None)
                data['rental_period'] = details.get('Kirayə dövrü', None)
                data['payment_form'] = details.get('Ödəniş forması', None)
                data['furnishing'] = details.get('Əmlakı', None)

                # Extract description
                desc_elem = soup.select_one('.umumi-melumat p')
                data['description'] = desc_elem.get_text(strip=True) if desc_elem else None

                # Extract title (usually from h1 or the main header)
                title_elem = soup.select_one('h1')
                data['title'] = title_elem.get_text(strip=True) if title_elem else None

                # Extract contact information
                contact_name_elem = soup.select_one('#elaq-shexs')
                if contact_name_elem:
                    contact_text = contact_name_elem.get_text(strip=True)
                    # Remove role suffix like "(Mülkiyyətçi)" or "(Agent)"
                    data['contact_name'] = re.sub(r'\s*\([^)]+\)\s*$', '', contact_text).strip()
                else:
                    data['contact_name'] = None

                # Extract phone number
                phone_elem = soup.select_one('#item-phone')
                data['phone'] = phone_elem.get_text(strip=True) if phone_elem else None

                # Extract email if present
                email_elem = soup.select_one('a[href^="mailto:"]')
                if email_elem:
                    email = email_elem.get('href', '').replace('mailto:', '')
                    data['email'] = email
                else:
                    data['email'] = None

                return data

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return {'url': url, 'listing_id': re.search(r'elan=(\d+)', url).group(1) if re.search(r'elan=(\d+)', url) else None, 'error': str(e)}

    async def scrape_all_listings(self, urls: List[str]) -> List[Dict]:
        """Scrape all listing details"""
        all_data = []

        # Process in batches
        batch_size = 20
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i+batch_size]
            tasks = [self.scrape_listing_details(url) for url in batch_urls]
            results = await asyncio.gather(*tasks)
            all_data.extend(results)

            print(f"Scraped {len(all_data)}/{len(urls)} listings...")
            await asyncio.sleep(1)  # Be polite to the server

        return all_data

    def save_to_csv(self, data: List[Dict], filename: str = 'tikili_listings.csv'):
        """Save data to CSV file"""
        if not data:
            print("No data to save")
            return

        keys = data[0].keys()
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

        print(f"Data saved to {filename}")

    def save_to_xlsx(self, data: List[Dict], filename: str = 'tikili_listings.xlsx'):
        """Save data to Excel file"""
        if not data:
            print("No data to save")
            return

        df = pd.DataFrame(data)
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"Data saved to {filename}")


async def main():
    print("Starting Tikili.az scraper...")

    async with TikiliScraper() as scraper:
        # Step 1: Get all listing URLs
        print("\nStep 1: Fetching listing URLs...")
        urls = await scraper.get_all_listing_urls(max_pages=50)  # Adjust max_pages as needed
        print(f"Found {len(urls)} unique listings")

        # Remove duplicates
        urls = list(set(urls))
        print(f"After removing duplicates: {len(urls)} listings")

        if not urls:
            print("No listings found!")
            return

        # Step 2: Scrape details from each listing
        print("\nStep 2: Scraping listing details...")
        data = await scraper.scrape_all_listings(urls)

        # Step 3: Save to files
        print("\nStep 3: Saving data...")
        scraper.save_to_csv(data)
        scraper.save_to_xlsx(data)

        print(f"\nDone! Scraped {len(data)} listings")
        print(f"Files saved: tikili_listings.csv and tikili_listings.xlsx")


if __name__ == "__main__":
    asyncio.run(main())