import urllib.request

def count_words_in_url(url):
    try:
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
        words = data.split()
        return len(words)
    except Exception as e:
        return f"Error accessing the URL: {e}"

def main():
    url = input("Enter a URL: ")
    result = count_words_in_url(url)
    print(result)

if __name__ == "__main__":
    main()