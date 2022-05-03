from bs4 import BeautifulSoup
import re
import requests

data = requests.get("http://www.gutenberg.org/cache/epub/8001/pg8001.html")
content = data.content

print(content[1100:2200])

# We can clearly see that the text is wrapped with HTML tags and making it difficult to analyse the content.

# We can strip the HTML tags using this strip_html_tags function.


def strip_html_tags(content):
    soup = BeautifulSoup(content, 'html.parser')
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    # It replaces all the newlines and tabs with newline.
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text


clean_text = strip_html_tags(content)
print(clean_text[1100:2200])
