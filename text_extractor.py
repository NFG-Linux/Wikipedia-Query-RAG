import requests                     # For making HTTP requests
from bs4 import BeautifulSoup       # For parsing HTML content

def scrape_webpage():
    # Hardcoded URL of the Rocket League Wikipedia page
    url = "https://en.wikipedia.org/wiki/Rocket_League"
    output_file = "Selected_Document.txt"  # File to store extracted text

    # Use a user-agent to simulate a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'
    }

    try:
        # Make the GET request to the page
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to retrieve page: HTTP {response.status_code}")
            return ""

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Select all <p> tags within the main content container (recursive)
        paragraphs = soup.select('div.mw-parser-output p')

        # Report how many paragraph tags were found
        print(f"Found {len(paragraphs)} paragraph tags.")

        # Remove empty or whitespace-only paragraphs
        filtered_paragraphs = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
        print(f"Filtered down to {len(filtered_paragraphs)} non-empty paragraphs.")

        if not filtered_paragraphs:
            print("No non-empty paragraph text found.")
            return ""

        # Join paragraphs with blank lines between them
        text = '\n\n'.join(filtered_paragraphs)

        # Write the extracted text to a file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"Successfully extracted and saved content to {output_file}")
        return text

    except Exception as e:
        # Handle network or parsing errors gracefully
        print(f"Error during scraping: {e}")
        return ""

def main():
    scrape_webpage()

# Run the script if executed directly
if __name__ == '__main__':
    main()
