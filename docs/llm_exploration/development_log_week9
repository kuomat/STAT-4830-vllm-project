# LLM Exploration Summary  

## Session Focus  
This week, we focused on scaling our dataset by adding product images, descriptions, and pricing data. Initially, we attempted to scrape product websites directly using Python tools and synthesized advice from LLMs (ChatGPT o1, Claude, DeepSeek). However, persistent anti-scraping measures on most major retail sites blocked these approaches. We then pivoted to locating ready-made datasets with complete product metadata. While many had image links, they often lacked associated product descriptions or prices. Eventually, we found datasets with direct image URLs and created a pipeline to automatically download and clean images, avoiding dynamic web scraping entirely. This was a key breakthrough in expanding our dataset efficiently.

## Surprising Insights  

### Conversation: Python-Based Web Scraping Strategies
**Prompt That Did Not Work:**  
- *"How do I extract product names and image URLs from H&M's listing pages using BeautifulSoup or Selenium?"*  

**Key Insights:**  
- H&M and similar e-commerce sites use JavaScript-based rendering, so standard requests + BeautifulSoup returned mostly empty pages.
- Selenium did not work smoothly in Google Colab, due to repeated ChromeDriver/Chromium version mismatches and lack of Snap support.
- Despite iterating with several LLMs (ChatGPT o1, Claude, DeepSeek), we were unable to get a fully working solution without extensive reverse engineering of each site's dynamic frontend logic.

**Prompt That Worked:**  
- *"Find me a Kaggle dataset with men's clothing images, prices, and descriptions similar to the women's ASOS dataset."*  

**Key Insights:**  
- Most publicly available fashion datasets are skewed toward women’s clothing, especially in curated platforms like Kaggle.  
- Datasets that contain high-quality images often lack prices or descriptions, and vice versa.
- While many JSON-based or image-heavy datasets exist (e.g., DeepFashion, OpenImages), finding ones with complete e-commerce metadata was rare.
- We pivoted toward datasets that include clean image URLs, which we could then process with a Python image downloader.  

## Techniques That Worked  
- Using direct image URLs in CSV files allowed us to bypass JavaScript-heavy retail pages.
- Switching from scraped HTML to dataset-driven pipelines gave us control over metadata consistency.
- Applying ast.literal_eval and split(',') helped normalize image lists stored as strings in different formats.
- Combining Pandas and Requests let us rapidly download and save thousands of product images.

## Dead Ends Worth Noting  

### Approach: Full Website Scraping via Selenium or BeautifulSoup 
- Repeated ChromeDriver version mismatches in Colab made automation frustrating and time-consuming.
- Even with matching versions, many sites blocked headless traffic or throttled requests aggressively.
- Trying to parse content from retail pages (e.g., SHEIN, Uniqlo) via static HTML failed because key content was loaded via JS.
- Multiple attempts using synthesized scraping templates from various LLMs didn’t resolve core access issues. 

### Approach: Relying on Prebuilt Fashion Datasets 
- Most available datasets didn’t include product descriptions or prices, making them insufficient for our use case.
- Even when text fields were present, they were often categorical tags or structured attributes, not rich product descriptions.
- We had to filter out several promising datasets because they lacked clean alignment between image and text fields.

## Next Steps  
- Continue expanding dataset by filtering and downloading from known-good CSVs with image, description, and price.
- Augment dataset with synthetic user preference vectors to simulate cold-start recommendation behavior.
- Benchmark CLIP and hybrid embeddings on this new dataset to explore clustering and similarity performance.
- Investigate ways to unify cleaned data from multiple clothing categories into a single recommendation-ready corpus.
- Re-run our models/algorithms with the agumented datasets.

### Questions to Explore  
- Are there lesser-known APIs or retail datasets with complete product metadata we haven’t discovered yet?
- Can we auto-generate structured product descriptions using LLMs from visual features and sparse titles?
