"""
Example usage of the HypothesisGenerator.
"""

from coscientist.hypothesis_generator import HypothesisGenerator


def main():
    # Initialize the generator
    generator = HypothesisGenerator()

    # Example inputs
    goal = """
    Find a mechanistic link between the RAB18 gene and metabolic dysfunction-associated 
    steatotic liver disease (MASLD). Consider whether inhibition of the RAB18 gene 
    would have a positive effect on a MASLD patient's health outcomes.
    """

    preferences = [
        "Clear mechanistic pathway linking RAB18 to MASLD",
        "Testable predictions about health outcomes",
        "Consideration of potential side effects",
        "Integration of existing literature",
    ]

    articles = """
    Recent studies have demonstrated that RAB18 plays a crucial role in lipid droplet 
    dynamics and metabolism. Key findings include:
    
    1. Smith et al. (2023) showed RAB18 regulates lipid droplet size
    2. Jones et al. (2022) linked RAB18 expression to fatty liver disease
    3. Zhang et al. (2023) demonstrated RAB18 inhibition reduces inflammation
    """

    # Generate hypothesis
    hypothesis = generator.generate_hypothesis(
        goal=goal, preferences=preferences, articles_with_reasoning=articles
    )

    # Print the result
    print("Generated Hypothesis:")
    print("-" * 80)
    print(hypothesis)


if __name__ == "__main__":
    main()
