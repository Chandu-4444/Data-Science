import unicodedata


def remove_accented_characters(text):
    """
    Remove accented characters from a string.
    """
    text = unicodedata.normalize('NFKD', text).encode(
        'ASCII', 'ignore').decode('utf-8', 'ignore')
    return text


text = remove_accented_characters('Sómě Áccěntěd těxt')

print(text)
