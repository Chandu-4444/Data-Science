from nltk.tokenize.toktok import ToktokTokenizer
import nltk
tokenizer = ToktokTokenizer()

stopword_list = nltk.corpus.stopwords.words('english')


def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    if is_lower_case:
        filtered_tokens = [
            token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [
            token for token in tokens if token.lower() not in stopword_list]

    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


print(remove_stopwords("The, and, if are stopwords, computer is not!"))
