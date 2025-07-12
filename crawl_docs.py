import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from appwrite_service import appwrite_service
import logging
import re
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

visited = set()
docs = []


def extract_links_from_url(base_url, domain):
    """Extract links from a documentation page"""
    try:
        res = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # Common selectors for documentation links
        link_selectors = [
            "a[href^='/']",  # Relative links
            "a[href^='./']",  # Relative links with dot
            "a[href^='../']",  # Parent directory links
            "nav a",  # Navigation links
            ".sidebar a",  # Sidebar links
            ".menu a",  # Menu links
            ".toc a",  # Table of contents links
        ]

        links = set()
        for selector in link_selectors:
            found_links = soup.select(selector)
            for link in found_links:
                href = link.get("href")
                if href:
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(base_url, href)
                    # Only include links from the same domain
                    if urlparse(absolute_url).netloc == urlparse(domain).netloc:
                        links.add(absolute_url)

        return list(links)
    except Exception as e:
        logger.error(f"Error extracting links from {base_url}: {str(e)}")
        return []


def extract_content_from_url(url):
    """Extract content from a documentation page"""
    if url in visited:
        return
    logger.info(f"Crawling: {url}")
    visited.add(url)

    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Try different content selectors for different documentation sites
        content_selectors = [
            "main",  # Common main content area
            "article",  # Article content
            ".content",  # Content class
            ".main-content",  # Main content class
            ".documentation",  # Documentation class
            ".doc-content",  # Doc content class
            "#content",  # Content ID
            "#main",  # Main ID
            "body",  # Fallback to body
        ]

        content_div = None
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                break

        if not content_div:
            logger.warning(f"No content found for {url}")
            return

        # Extract text from various elements
        text_elements = content_div.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",  # Headers
                "p",  # Paragraphs
                "pre",
                "code",  # Code blocks
                "li",  # List items
                "div",  # Divs (with some filtering)
                "section",  # Sections
            ]
        )

        # Filter and clean text
        text_parts = []
        for element in text_elements:
            text = element.get_text().strip()
            if text and len(text) > 10:  # Only include substantial text
                text_parts.append(text)

        text = "\n".join(text_parts)

        if text and len(text) > 50:  # Only save if there's substantial content
            title = soup.title.string if soup.title else "No Title"
            docs.append(
                {
                    "url": url,
                    "title": title,
                    "content": text,
                }
            )
            logger.info(f"Extracted content from {url}: {len(text)} characters")
        else:
            logger.warning(f"Insufficient content from {url}")

    except Exception as e:
        logger.error(f"Error at {url}: {str(e)}")


def crawl_documentation(base_url):
    """Crawl documentation from a given URL"""
    global visited, docs

    # Reset state
    visited = set()
    docs = []

    try:
        logger.info(f"Starting documentation crawl for: {base_url}")

        # Check if documentation already exists
        if appwrite_service.docs_already_exist(base_url):
            logger.info(f"Documentation for {base_url} already exists. Skipping crawl.")
            return True

        # Parse the base URL to get domain
        parsed_url = urlparse(base_url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Extract links from the base URL
        links = extract_links_from_url(base_url, domain)
        logger.info(f"Found {len(links)} links to crawl")

        # Also add the base URL itself
        links.append(base_url)

        # Limit the number of pages to crawl to avoid overwhelming the server
        max_pages = 50
        links = links[:max_pages]

        # Crawl each link
        for i, link in enumerate(links):
            logger.info(f"Crawling {i+1}/{len(links)}: {link}")
            extract_content_from_url(link)

            # Add a small delay to be respectful to the server
            import time

            time.sleep(0.5)

        logger.info(f"Crawled {len(docs)} documents")

        if not docs:
            logger.error("No documents were successfully crawled")
            return False

        # Save to Appwrite storage with URL
        success = appwrite_service.save_raw_docs_to_storage(docs, base_url)

        if success:
            logger.info("Successfully saved raw documents to Appwrite storage")
            return True
        else:
            logger.error("Failed to save raw documents to Appwrite storage")
            return False

    except Exception as e:
        logger.error(f"Error in crawl_documentation: {str(e)}")
        return False


def crawl_and_save_docs():
    """Legacy function for React docs - kept for backward compatibility"""
    return crawl_documentation("https://react.dev/learn")


if __name__ == "__main__":
    # Get URL from command line argument if provided
    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"🕷️  Crawling documentation for URL: {url}")
    else:
        url = "https://react.dev/learn"  # Default URL
        print(f"🕷️  Crawling documentation for default URL: {url}")

    success = crawl_documentation(url)
    if success:
        print("✅ Crawling completed successfully!")
    else:
        print("❌ Crawling failed!")
