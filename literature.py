import requests
import time
import re
from bs4 import BeautifulSoup


class LiteratureAgent:
    def __init__(self):
        self.base_url = "https://api.openalex.org/works"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Scientific Research Agent; materials-science-bot)",
            "From": "researcher@example.com",
        }
        self.browser_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def _reconstruct_abstract(self, inverted_index):
        if not inverted_index:
            return None
        word_list = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_list.append((pos, word))
        word_list.sort(key=lambda x: x[0])
        return " ".join([w[1] for w in word_list])

    def _fetch_html_text(self, url):
        try:
            print(f"      ☁️ Fetching full text from: {url[:50]}...")
            resp = requests.get(url, headers=self.browser_headers, timeout=30)
            if resp.status_code != 200:
                return None
            soup = BeautifulSoup(resp.content, "html.parser")
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text(separator=" ")
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = "\n".join(chunk for chunk in chunks if chunk)
            if len(clean_text) < 300:
                return None
            return clean_text
        except Exception:
            return None

    def _try_unpaywall(self, doi):
        """Non-OA fallback: probe for legitimate OA full-text links by DOI via Unpaywall; return None on failure without blocking"""
        if not doi:
            return None
        try:
            url = f"https://api.unpaywall.org/v2/{doi}?email=researcher@example.com"
            resp = requests.get(url, headers=self.browser_headers, timeout=30)
            if resp.status_code != 200:
                return None
            data = resp.json()
            loc = data.get("best_oa_location") or {}
            return loc.get("url_for_pdf") or loc.get("url") or None
        except Exception:
            return None

    def enrich_papers_with_fulltext(self, papers, max_fulltext=10):
        count = 0
        for p in papers:
            if count >= max_fulltext:
                break
            oa_url = p.get("open_access", {}).get("oa_url")
            if not oa_url and not p.get("full_text"):
                oa_url = self._try_unpaywall(p.get("doi"))
            if oa_url and not p.get("full_text"):
                full_text = self._fetch_html_text(oa_url)
                if full_text:
                    p["full_text"] = full_text
                    p["has_full_text"] = True
                    count += 1
                else:
                    p["has_full_text"] = False
            else:
                p["has_full_text"] = False
        print(f"   📥 Successfully retrieved full text for {count} papers.")
        return papers

    def search_literature(self, query, limit=10):
        q1 = f"{query} microwave dielectric properties"
        params1 = {
            "search": q1,
            "per_page": limit,
            "filter": "has_abstract:true,type:article",
            "sort": "relevance_score:desc",
        }
        q2 = f"{query} dielectric constant ceramic"
        params2 = {
            "search": q2,
            "per_page": limit,
            "filter": "has_abstract:true,type:article",
            "sort": "relevance_score:desc",
        }

        def _fetch(p):
            try:
                r = requests.get(
                    self.base_url, params=p, headers=self.headers, timeout=15
                )
                if r.status_code == 200:
                    return r.json().get("results", [])
                return []
            except:
                return []

        r1 = _fetch(params1)
        r2 = _fetch(params2)

        results = []
        seen_ids = set()
        for p in r1 + r2:
            if p["id"] not in seen_ids:
                results.append(p)
                seen_ids.add(p["id"])
        return results

    def online_search(self, query, limit=5):
        params = {
            "search": query,
            "per_page": limit,
            "filter": "has_abstract:true",
            "sort": "relevance_score:desc",
        }
        try:
            r = requests.get(
                self.base_url, params=params, headers=self.headers, timeout=15
            )
            if r.status_code == 200:
                results = r.json().get("results", [])
                processed = self.process_papers_to_dicts(results)
                return processed
            return []
        except Exception as e:
            print(f"   [!] Online search error: {e}")
            return []

    def process_papers_to_dicts(self, papers):
        if not papers:
            return []
        processed_list = []
        for p in papers:
            title = p.get("title", "Untitled").replace("\n", " ")
            year = str(p.get("publication_year", "n.d."))
            is_full = p.get("has_full_text", False)
            journal = (
                p.get("primary_location", {})
                .get("source", {})
                .get("display_name", "OpenAlex Database")
            )
            if not journal:
                journal = "OpenAlex Database"
            authors_list = [
                a.get("author", {}).get("display_name", "")
                for a in p.get("authorships", [])
            ]
            content = ""
            if is_full:
                raw_text = p["full_text"]
                if len(raw_text) > 8000:
                    content = raw_text[:5000] + "\n...[Skipped]...\n" + raw_text[-3000:]
                else:
                    content = raw_text
                content = "[FULL TEXT AVAILABLE] " + content
            else:
                abstract = (
                    p.get("abstract")
                    or self._reconstruct_abstract(p.get("abstract_inverted_index"))
                    or "(No abstract)"
                )
                content = "[Abstract Only] " + abstract
            processed_list.append(
                {
                    "title": title,
                    "authors": authors_list,
                    "year": year,
                    "journal": journal,
                    "content": content,
                    "source": p.get("doi", p.get("id", "OpenAlex")),
                    "type": "Web",
                    "score": 2.0,
                }
            )
        return processed_list

    def get_citation_list(self, papers):
        return []
