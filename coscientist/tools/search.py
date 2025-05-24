"""
Search tools for literature review and information gathering.
"""

import json
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass
class SearchResult:
    """Represents a search result from web search."""
    title: str
    url: str
    snippet: str
    source: str = "web"
    relevance_score: float = 0.0


class WebSearchTool:
    """
    Web search tool for finding relevant scientific literature and information.
    
    Note: This is a simplified implementation. In production, you would
    integrate with services like Google Scholar API, PubMed API, etc.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.rate_limit_delay = 1.0  # Seconds between requests
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement simple rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def search_academic_papers(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Search for academic papers and scientific literature.
        
        Parameters
        ----------
        query: str
            Search query
        max_results: int
            Maximum number of results to return
            
        Returns
        -------
        List[SearchResult]
            List of search results
        """
        self._rate_limit()
        
        # Mock implementation - in production, integrate with actual APIs
        # such as Google Scholar, PubMed, arXiv, etc.
        mock_results = [
            SearchResult(
                title=f"Research paper on {query} - Study {i+1}",
                url=f"https://example.com/paper_{i+1}",
                snippet=f"This paper investigates {query} and presents novel findings about molecular mechanisms...",
                source="academic",
                relevance_score=0.9 - (i * 0.1)
            )
            for i in range(min(max_results, 5))
        ]
        
        return mock_results
    
    def search_general_web(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Search the general web for information.
        
        Parameters
        ----------
        query: str
            Search query
        max_results: int
            Maximum number of results to return
            
        Returns
        -------
        List[SearchResult]
            List of search results
        """
        self._rate_limit()
        
        # Mock implementation - in production, integrate with search APIs
        mock_results = [
            SearchResult(
                title=f"Information about {query} - Result {i+1}",
                url=f"https://example.com/result_{i+1}",
                snippet=f"General information about {query} including background, mechanisms, and applications...",
                source="web",
                relevance_score=0.8 - (i * 0.1)
            )
            for i in range(min(max_results, 3))
        ]
        
        return mock_results
    
    def search_patents(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search for patents related to the query.
        
        Parameters
        ----------
        query: str
            Search query
        max_results: int
            Maximum number of results to return
            
        Returns
        -------
        List[SearchResult]
            List of patent search results
        """
        self._rate_limit()
        
        # Mock implementation - in production, integrate with patent databases
        mock_results = [
            SearchResult(
                title=f"Patent: Novel methods for {query} - Patent {i+1}",
                url=f"https://patents.example.com/patent_{i+1}",
                snippet=f"This patent describes innovative approaches to {query} with applications in biotechnology...",
                source="patent",
                relevance_score=0.7 - (i * 0.1)
            )
            for i in range(min(max_results, 2))
        ]
        
        return mock_results


class LiteratureReviewTool:
    """
    Tool for conducting comprehensive literature reviews using search and LLM analysis.
    """
    
    def __init__(self, search_tool: WebSearchTool, llm: BaseChatModel):
        self.search_tool = search_tool
        self.llm = llm
    
    def conduct_literature_review(self, research_goal: str, key_terms: List[str]) -> Dict[str, Any]:
        """
        Conduct a comprehensive literature review.
        
        Parameters
        ----------
        research_goal: str
            The research goal or topic
        key_terms: List[str]
            Key terms to search for
            
        Returns
        -------
        Dict[str, Any]
            Literature review results
        """
        all_results = []
        
        # Search for each key term
        for term in key_terms:
            # Academic papers
            academic_results = self.search_tool.search_academic_papers(term, max_results=5)
            all_results.extend(academic_results)
            
            # General web
            web_results = self.search_tool.search_general_web(term, max_results=3)
            all_results.extend(web_results)
            
            # Patents
            patent_results = self.search_tool.search_patents(term, max_results=2)
            all_results.extend(patent_results)
        
        # Remove duplicates and sort by relevance
        unique_results = self._deduplicate_results(all_results)
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Analyze results with LLM
        analysis = self._analyze_literature_with_llm(research_goal, unique_results[:15])
        
        return {
            "research_goal": research_goal,
            "key_terms": key_terms,
            "total_results": len(unique_results),
            "top_results": unique_results[:10],
            "analysis": analysis,
            "generated_at": time.time()
        }
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL."""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    def _analyze_literature_with_llm(self, research_goal: str, results: List[SearchResult]) -> str:
        """
        Analyze literature results using LLM.
        
        Parameters
        ----------
        research_goal: str
            Research goal
        results: List[SearchResult]
            Search results to analyze
            
        Returns
        -------
        str
            LLM analysis of the literature
        """
        # Prepare literature summary for LLM
        literature_summary = "\n\n".join([
            f"Title: {result.title}\nSource: {result.source}\nSummary: {result.snippet}"
            for result in results[:10]  # Limit to top 10 for LLM context
        ])
        
        analysis_prompt = f"""
        You are conducting a literature review for the following research goal:
        
        Research Goal: {research_goal}
        
        Based on the following literature findings, provide a comprehensive analysis:
        
        Literature Sources:
        {literature_summary}
        
        Please provide:
        1. Key themes and trends in the literature
        2. Gaps in current knowledge
        3. Methodological approaches being used
        4. Conflicting findings or controversies
        5. Opportunities for novel research
        6. Most relevant and high-impact sources
        
        Structure your response as a professional literature review section.
        """
        
        try:
            response = self.llm.invoke(analysis_prompt)
            return response.content
        except Exception as e:
            return f"Error analyzing literature: {str(e)}"


class FactCheckingTool:
    """
    Tool for fact-checking claims and hypotheses against known scientific literature.
    """
    
    def __init__(self, search_tool: WebSearchTool, llm: BaseChatModel):
        self.search_tool = search_tool
        self.llm = llm
    
    def fact_check_hypothesis(self, hypothesis: str) -> Dict[str, Any]:
        """
        Fact-check a hypothesis against available literature.
        
        Parameters
        ----------
        hypothesis: str
            Hypothesis to fact-check
            
        Returns
        -------
        Dict[str, Any]
            Fact-checking results
        """
        # Extract key claims from hypothesis
        key_claims = self._extract_key_claims(hypothesis)
        
        fact_check_results = []
        
        for claim in key_claims:
            # Search for supporting/contradicting evidence
            search_results = self.search_tool.search_academic_papers(claim, max_results=5)
            
            # Analyze evidence
            evidence_analysis = self._analyze_evidence(claim, search_results)
            
            fact_check_results.append({
                "claim": claim,
                "evidence": search_results,
                "analysis": evidence_analysis
            })
        
        # Overall assessment
        overall_assessment = self._generate_overall_assessment(hypothesis, fact_check_results)
        
        return {
            "hypothesis": hypothesis,
            "key_claims": key_claims,
            "fact_check_results": fact_check_results,
            "overall_assessment": overall_assessment,
            "confidence_score": self._calculate_confidence_score(fact_check_results)
        }
    
    def _extract_key_claims(self, hypothesis: str) -> List[str]:
        """Extract key factual claims from a hypothesis."""
        extraction_prompt = f"""
        Extract the key factual claims from the following hypothesis that can be verified against scientific literature:
        
        Hypothesis: {hypothesis}
        
        Return a list of specific, verifiable claims (3-5 claims maximum).
        Format as a JSON list of strings.
        """
        
        try:
            response = self.llm.invoke(extraction_prompt)
            claims = json.loads(response.content)
            if isinstance(claims, list):
                return claims[:5]  # Limit to 5 claims
        except:
            pass
        
        # Fallback: use the entire hypothesis as a single claim
        return [hypothesis]
    
    def _analyze_evidence(self, claim: str, search_results: List[SearchResult]) -> str:
        """Analyze evidence for a specific claim."""
        evidence_text = "\n\n".join([
            f"Source: {result.title}\nContent: {result.snippet}"
            for result in search_results
        ])
        
        analysis_prompt = f"""
        Analyze the following evidence for the claim:
        
        Claim: {claim}
        
        Evidence:
        {evidence_text}
        
        Provide an assessment of whether the evidence supports, contradicts, or is neutral regarding the claim.
        Include a confidence level (high/medium/low) and brief reasoning.
        """
        
        try:
            response = self.llm.invoke(analysis_prompt)
            return response.content
        except Exception as e:
            return f"Error analyzing evidence: {str(e)}"
    
    def _generate_overall_assessment(self, hypothesis: str, fact_check_results: List[Dict]) -> str:
        """Generate overall assessment of hypothesis validity."""
        results_summary = "\n\n".join([
            f"Claim: {result['claim']}\nAnalysis: {result['analysis']}"
            for result in fact_check_results
        ])
        
        assessment_prompt = f"""
        Based on the fact-checking results below, provide an overall assessment of this hypothesis:
        
        Hypothesis: {hypothesis}
        
        Fact-checking Results:
        {results_summary}
        
        Provide:
        1. Overall validity assessment (strongly supported/supported/neutral/contradicted/strongly contradicted)
        2. Key strengths of the hypothesis
        3. Key weaknesses or contradictions
        4. Recommendations for further investigation
        """
        
        try:
            response = self.llm.invoke(assessment_prompt)
            return response.content
        except Exception as e:
            return f"Error generating assessment: {str(e)}"
    
    def _calculate_confidence_score(self, fact_check_results: List[Dict]) -> float:
        """Calculate a confidence score for the fact-checking."""
        # Simple heuristic based on number of claims and evidence quality
        if not fact_check_results:
            return 0.0
        
        # Score based on number of claims verified
        base_score = min(len(fact_check_results) / 5.0, 1.0)
        
        # Adjust based on evidence quality (mock calculation)
        evidence_quality = sum(len(result['evidence']) for result in fact_check_results) / len(fact_check_results)
        quality_factor = min(evidence_quality / 5.0, 1.0)
        
        return base_score * quality_factor


# Factory function for creating search tools
def create_search_tools(llm: BaseChatModel, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Create and configure search tools.
    
    Parameters
    ----------
    llm: BaseChatModel
        Language model for analysis
    api_key: Optional[str]
        API key for search services
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of configured search tools
    """
    web_search = WebSearchTool(api_key=api_key)
    literature_review = LiteratureReviewTool(web_search, llm)
    fact_checker = FactCheckingTool(web_search, llm)
    
    return {
        "web_search": web_search,
        "literature_review": literature_review,
        "fact_checker": fact_checker
    }
