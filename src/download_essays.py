import requests
import os
import time
from html.parser import HTMLParser
import json

ESSAYS = [
    ("do_things_that_dont_scale.txt",        "https://paulgraham.com/ds.html"),
    ("how_to_get_startup_ideas.txt",          "https://paulgraham.com/startupideas.html"),
    ("what_i_worked_on.txt",                  "https://paulgraham.com/worked.html"),
    ("keep_your_identity_small.txt",          "https://paulgraham.com/identity.html"),
    ("the_anatomy_of_determination.txt",      "https://paulgraham.com/determination.html"),
    ("what_youll_wish_youd_known.txt",        "https://paulgraham.com/hs.html"),
    ("how_to_do_what_you_love.txt",           "https://paulgraham.com/love.html"),
    ("makers_schedule_managers_schedule.txt", "https://paulgraham.com/makersschedule.html"),
    ("default_alive_or_default_dead.txt",     "https://paulgraham.com/aord.html"),
    ("hiring_is_obsolete.txt",                "https://paulgraham.com/hiring.html"),
]

class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {"script", "style"}
        self.current_skip = False

    def handle_starttag(self, tag, attributes):
        if tag in self.skip_tags:
            self.current_skip = True
    
    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.current_skip = False

    def handle_data(self, data):
        if not self.current_skip:
            self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts)
    
def fetch_essay(url):
    headers = {"User-Agent": "rag-pipeline-task/1.0 (learning project)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    parser = HTMLTextExtractor()
    parser.feed(r.text)
    text = parser.get_text()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size-overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def download_all():
    os.makedirs("data/essays", exit_ok=True)
    all_chunks = []
    total_chunks = 0

    for filename, url in ESSAYS:
        print(f"Fetching: {filename}")
        try:
            text = fetch_essay(url)
            words = len(text.split())
            print(f" -> {words} words")

            filepath = os.path.join("data", "essays", filename)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)

            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append({
                    "essay": filename.replace(".txt", ""),
                    "text": chunk
                })
            total_chunks += len(chunks)
            print(f" -> {len(chunks)} chunks")

        except Exception as e:
            print(f" Error: {e}")

        time.sleep(1)

    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(ESSAYS)} essays, {total_chunks} chunks saved to data/chunks.json")

if __name__=="__name__":
    download_all()