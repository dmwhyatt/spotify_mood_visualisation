#!/usr/bin/env python3
"""
Test Genius API credentials are working.
"""

import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logger = logging.getLogger(__name__)


def test_genius_token():
    """Test if Genius token is valid."""
    logger.info("Genius API token test")

    token = (
        os.getenv("GENIUS_CLIENT_ACCESS_TOKEN")
        or os.getenv("GENIUS_ACCESS_TOKEN")
        or os.getenv("GENIUS_CLIENT_SECRET")
    )

    if not token:
        logger.error(
            "No token in .env. Set GENIUS_CLIENT_ACCESS_TOKEN (from "
            "https://genius.com/api-clients, Client Access Token)."
        )
        return False

    logger.info("Token loaded (%s chars): %s...%s", len(token), token[:8], token[-4:])

    try:
        import requests

        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36"
            ),
        }

        response = requests.get(
            "https://api.genius.com/search",
            params={"q": "Pharrell Williams Happy"},
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()

        hits = response.json().get("response", {}).get("hits", [])

        if not hits:
            logger.warning("API OK but no search hits (unusual); token may still be fine.")
            return True

        song = hits[0]["result"]
        title = song["title"]
        artist = song["primary_artist"]["name"]
        url = song["url"]
        logger.info("API OK: %s - %s", artist, title)

        from bs4 import BeautifulSoup

        page = requests.get(url, headers={"User-Agent": headers["User-Agent"]})
        soup = BeautifulSoup(page.text, "html.parser")
        lyrics_divs = soup.find_all("div", {"data-lyrics-container": "true"})

        if lyrics_divs:
            preview = lyrics_divs[0].get_text()[:80].strip()
            logger.info("Lyrics scrape OK (preview): %s...", preview)
        else:
            logger.warning(
                "Lyrics containers not found on page (Genius HTML may have changed)."
            )

        return True

    except Exception as e:
        err = str(e)
        logger.error("Test failed: %s", err)

        if "403" in err or "Forbidden" in err:
            logger.error(
                "403: use Client Access Token from api-clients, not ID/secret; "
                "retry after a few minutes if rate-limited."
            )
        elif "401" in err or "Unauthorized" in err:
            logger.error("401: token invalid or expired — generate a new one at genius.com/api-clients.")
        else:
            logger.error("Check network and dependencies (requests, beautifulsoup4).")

        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(0 if test_genius_token() else 1)
