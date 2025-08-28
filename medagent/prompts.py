multiquery_prompt = """Generate {num_queries} comprehensive research queries in natural English for: {query}

Each query should be written as if searching a research database in plain language, targeting different dimensions of the research topic:

COVERAGE AREAS:
- Core applications and use cases
- Recent developments and innovations (2019-2025)
- Comparative studies and performance evaluations
- Clinical implementations and real-world applications
- Methodological approaches and technical frameworks
- Outcomes, efficacy, and impact assessments
- Limitations, challenges, and future directions
- Different medical specialties or subdomain applications

QUERY CHARACTERISTICS:
- Use natural medical and scientific terminology
- Include relevant time periods when appropriate
- Focus on peer-reviewed research and clinical studies
- Cover both broad overviews and specific implementations
- Consider different study types (reviews, trials, case studies)
- Include emerging trends and established practices

Each query should explore a distinct angle while maintaining scientific rigor and searchability.

Return only the search queries as a numbered list."""

pubmed_parser_prompt = """
Parse the following natural language query for a PubMed literature search and extract structured information.

Query: "{natural_query}"

EXTRACTION RULES:
1. AUTHORS:
  - Extract full names exactly as written (e.g., "Anthony Fauci", "John Smith")
  - Handle multiple authors: "Smith and Jones" → authors: ["Smith", "Jones"]
  - Include variations: "Fauci" or "Anthony Fauci" → authors: ["Anthony Fauci"]

2. TOPICS:
  - Extract complete phrases as single topics, don't split compound concepts
  - Keep medical/scientific terms, diseases, treatments as coherent units
  - Only add synonyms for well-known medical abbreviations
  - Examples:
    * "effects of the Gaza war on children" → topics: ["effects of the Gaza war on children"]
    * "COVID treatment" → topics: ["COVID treatment", "COVID-19 treatment"]
    * "breast cancer screening" → topics: ["breast cancer screening"]
    * "RNA studies" → topics: ["RNA studies"]

3. DATE PARSING:
  - Convert relative terms to absolute dates (current year: 2025)
  - "last 5 years" → start_date: "2020/01/01", end_date: "2025/12/31"
  - "recent studies" → start_date: "2022/01/01", end_date: "2025/12/31"
  - "past decade" → start_date: "2015/01/01", end_date: "2025/12/31"
  - "2015 to 2020" → start_date: "2015/01/01", end_date: "2020/12/31"
  - "since 2020" → start_date: "2020/01/01", end_date: "2025/12/31"
  - If no dates mentioned → start_date: "", end_date: ""

4. MAX_RESULTS:
  - Extract explicit numbers: "100 papers" → max_results: 100
  - Handle qualitative terms: "many studies" → max_results: 100, "few papers" → max_results: 25
  - Default to 50 unless specified

5. FILENAME:
  - Always prepend with "data/" directory
  - Generate descriptive, filesystem-safe names
  - Format: data/main_topic_authors_timeframe.csv
  - Replace spaces with underscores, remove special characters
  - Truncate long names to keep under 60 characters total

EXAMPLES:
- "papers by Fauci on COVID from last 3 years" → authors: ["Fauci"], topics: ["COVID"], start_date: "2022/01/01", end_date: "2025/12/31", filename: "data/covid_fauci_2022-2025.csv"
- "cardiovascular research from 2015 to 2020" → authors: [], topics: ["cardiovascular research"], start_date: "2015/01/01", end_date: "2020/12/31", filename: "data/cardiovascular_research_2015-2020.csv"
- "RNA studies by Holland and Oz" → authors: ["Holland", "Oz"], topics: ["RNA studies"], start_date: "", end_date: "", filename: "data/rna_studies_holland_oz.csv"
- "effects of the Gaza war on children from 2015 to 2025" → authors: [], topics: ["effects of the Gaza war on children"], start_date: "2015/01/01", end_date: "2025/12/31", filename: "data/gaza_war_effects_children_2015-2025.csv"
- "systematic reviews on diabetes treatment" → authors: [], topics: ["systematic reviews on diabetes treatment"], start_date: "", end_date: "", filename: "data/systematic_reviews_diabetes_treatment.csv"
- "breast cancer screening in elderly women" → authors: [], topics: ["breast cancer screening in elderly women"], start_date: "", end_date: "", filename: "data/breast_cancer_screening_elderly.csv"
"""
